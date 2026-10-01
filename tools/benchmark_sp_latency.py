#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast

from mmseg.structures import SegDataSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure synchronized spatial-parallel model inference latency.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--output-dir', required=True, help='Directory for manifest and raw records')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--scene-img', help='Single original-scene image to preload before timing')
    parser.add_argument('--nirrgb-to-rgb', action='store_true', help='Use channels 1, 2, and 3 from a four-channel NIR-RGB image')
    parser.add_argument('--save-predictions', action='store_true', help='Save rank-0 predicted labels outside the timing result')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if args.warmup < 0 or args.samples < 1:
        raise ValueError('warmup must be non-negative and samples must be positive')
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_info() -> list[dict[str, Any]]:
    return [
        {
            'index': index,
            'name': torch.cuda.get_device_name(index),
            'memory_bytes': torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]


def visible_nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def load_scene_batch(path: Path, nirrgb_to_rgb: bool) -> dict[str, Any]:
    image = np.asarray(Image.open(path))
    if nirrgb_to_rgb:
        if image.ndim != 3 or image.shape[2] < 4:
            raise ValueError('--nirrgb-to-rgb requires an image with at least four channels')
        image = image[:, :, 1:4]
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected an RGB image after conversion, got shape {image.shape}')
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    height, width = image.shape[:2]
    sample = SegDataSample()
    sample.set_metainfo({
        'img_path': str(path),
        'ori_shape': (height, width),
        'img_shape': (height, width),
        'pad_shape': (height, width),
        'scale_factor': (1.0, 1.0),
        'flip': False,
        'flip_direction': None,
        'padding_size': [0, 0, 0, 0],
    })
    return {'inputs': [tensor], 'data_samples': [sample]}


def synchronized_latency(model: torch.nn.Module, data_batch: dict[str, Any], device: torch.device, fp16: bool) -> tuple[float, float]:
    # Timing excludes dataloader waiting; preprocessing through result broadcast is included.
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), autocast(enabled=fp16):
        model.test_step(data_batch)
    torch.cuda.synchronize(device)
    local_ms = (time.perf_counter() - start) * 1000.0
    duration = torch.tensor([local_ms], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        dist.barrier()
    return local_ms, float(duration.item())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.launcher == 'pytorch' and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    if not dist.is_initialized():
        raise RuntimeError('Launch with torch distributed; spatial-parallel inference requires a process group')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    cfg = Config.fromfile(config_path)
    cfg.launcher = 'pytorch'
    cfg.work_dir = str(output_dir / 'runner_work_dir')
    cfg.load_from = str(checkpoint_path)
    cfg.resume = False
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 1
    cfg.test_dataloader.persistent_workers = False
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()

    batches: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    if args.scene_img:
        scene_path = Path(args.scene_img).resolve()
        if rank == 0 and not scene_path.exists():
            raise FileNotFoundError(scene_path)
        dist.barrier()
        batch = load_scene_batch(scene_path, args.nirrgb_to_rgb)
        batches = [batch for _ in range(args.warmup + args.samples)]
        sample_ids = [str(scene_path)] * len(batches)
    else:
        dataloader = runner.test_dataloader
        iterator = iter(dataloader)
        for _ in range(args.warmup + args.samples):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            batches.append(batch)
            data_samples = batch.get('data_samples', [])
            meta = data_samples[0].metainfo if data_samples else {}
            sample_ids.append(str(meta.get('img_path', len(sample_ids))))

    for batch in batches[:args.warmup]:
        synchronized_latency(model, batch, device, bool(cfg.test_cfg.get('fp16', False)))

    records: list[dict[str, Any]] = []
    predictions_dir = output_dir / 'predictions'
    if args.save_predictions and rank == 0:
        predictions_dir.mkdir()
    if args.save_predictions:
        dist.barrier()
    for index, batch in enumerate(batches[args.warmup:]):
        local_ms, max_ms = synchronized_latency(model, batch, device, bool(cfg.test_cfg.get('fp16', False)))
        data_sample = batch['data_samples'][0]
        meta = data_sample.metainfo
        if args.save_predictions:
            with torch.no_grad(), autocast(enabled=bool(cfg.test_cfg.get('fp16', False))):
                prediction = model.test_step(batch)[0].pred_sem_seg.data.squeeze().detach().cpu().numpy()
            if rank == 0:
                np.save(predictions_dir / f'{index:05d}.npy', prediction)
        records.append({
            'sample_index': index,
            'sample_id': sample_ids[args.warmup + index],
            'rank': rank,
            'world_size': world_size,
            'local_ms': local_ms,
            'max_rank_ms': max_ms,
            'input_shape': list(meta.get('ori_shape', meta.get('img_shape', ()))),
        })

    gathered: list[list[dict[str, Any]] | None] = [None] * world_size if rank == 0 else []
    dist.gather_object(records, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        max_records = [record for record in records]
        values = np.asarray([record['max_rank_ms'] for record in max_records], dtype=np.float64)
        manifest = {
            'benchmark': 'synchronized spatial-parallel model inference latency',
            'timing_boundary': 'includes model preprocessing, device transfer, spatial partitioning, model computation, SP communication, rank-0 reconstruction, and result broadcast; excludes decoding, dataloader waiting, metrics, visualization, output writing, and benchmark coordination',
            'world_size': world_size,
            'spatial_layout': '1x2 for world_size=2; 2x2 for world_size=4; implementation rank order is column-major',
            'config': str(config_path),
            'config_sha256': sha256(config_path),
            'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': sha256(checkpoint_path),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'nccl': torch.cuda.nccl.version(),
            'gpus': gpu_info(),
            'nvidia_smi': visible_nvidia_smi(),
            'warmup_samples': args.warmup,
            'timed_samples': args.samples,
            'scene_img': str(Path(args.scene_img).resolve()) if args.scene_img else None,
            'nirrgb_to_rgb': args.nirrgb_to_rgb,
            'median_ms': float(np.median(values)),
            'mean_ms': float(np.mean(values)),
            'std_ms': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'p10_ms': float(np.percentile(values, 10)),
            'p90_ms': float(np.percentile(values, 90)),
        }
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (output_dir / 'rank_records.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            for rank_records in gathered:
                writer.writerows(rank_records or [])
        print(json.dumps(manifest, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
