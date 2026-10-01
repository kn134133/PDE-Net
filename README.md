# PDE-Net: Preserving Spatial Continuity and Fine-Scale Details in End-to-End Semantic Segmentation of Large-Swath Remote Sensing Imagery

PDE-Net is a research implementation for semantic segmentation of large-swath remote sensing imagery. It combines spatial-parallel whole-image processing with positional and detail-preserving modules so that a large scene can be partitioned across GPUs without treating each region as an independent crop.

The implementation is built on a local MMSegmentation-derived codebase and uses a Swin Transformer-Large backbone with a UPerNet decoder. It contains the configurations and core modules used for experiments on **GLH-Water**, **UAVid**, and **GF2-GID-15**.

> **Release scope.** This repository distributes source code and representative configurations only. Datasets, pretrained checkpoints, benchmark artifacts, manuscript files, and visualization figures are not included.

## Why PDE-Net?

Conventional remote-sensing segmentation commonly divides an original scene into fixed crops, predicts each crop independently, and then merges the results. This makes large images tractable, but it can weaken continuity at region boundaries and make narrow structures, small objects, and fine class transitions harder to preserve.

PDE-Net targets these limitations with three complementary components:

1. **Spatial-parallel processing.** The full image is spatially partitioned among distributed processes. The spatial-parallel Swin backbone exchanges features across regions, allowing the model to use cross-region context during whole-image processing.
2. **Cross-region positional encoding.** `SwinTransformer_sp_layer_cpe` supports sinusoidal cross-region positional encoding (CPE), enabling the backbone to distinguish the global locations of tokens that originate from different image partitions.
3. **Detail-aware decoding.** The released decoder variants include a Spatial Information Network (SIN) and High-Resolution Detail Enhancement (HRDE) modules. SIN combines spatial and channel feature refinement, while HRDE uses edge-aware and multi-scale detail refinement with gated residual enhancement.

The four-GPU experiments use a 2 × 2 spatial partition. This is a distributed implementation choice for processing a large image; all reported end-to-end latency values below are measured in that four-GPU setting.

## Experimental results

### Speed and accuracy comparison

The table below compares conventional crop-based processing at 1024 × 1024 with PDE-Net end-to-end inference. Crop-based times are the existing sliding-window measurements. PDE-Net times are newly measured four-GPU end-to-end latencies under each dataset's original-image or full-scene inference setting.

| Processing mode | Inference setting | GLH-Water | UAVid | GF2-GID-15 |
| --- | --- | --- | --- | --- |
| Crop-based | 1024 × 1024 | 18.98 s/image; 87.52 Water IoU | 1.54 s/image; 71.27 mIoU | 11.18 s/image; 70.17 mIoU |
| PDE-Net | Whole image, 4 GPUs | **8.61 s/image; 90.32 Water IoU** | **0.765 s/image; 73.19 mIoU** | **2.10 s/image; 83.15 mIoU** |

These results show lower reported inference latency and higher reported accuracy for PDE-Net than the strongest listed crop-based result for each dataset. The accuracy metrics follow the conventions of each dataset: Water IoU for GLH-Water and mIoU for UAVid and GF2-GID-15.

### Experimental protocol

- **Hardware:** four NVIDIA Tesla V100-series GPUs with 32 GB memory per GPU.
- **Backbone and decoder:** ImageNet-22K-pretrained Swin Transformer-Large with a UPerNet decoder in the reported experiments.
- **Optimizer and schedule:** AdamW with a polynomial learning-rate schedule.
- **Timing:** the benchmark synchronizes CUDA before and after each `model.test_step`, uses the maximum process latency as the distributed latency, and excludes data-loader waiting. See [`tools/benchmark_sp_latency.py`](tools/benchmark_sp_latency.py).
- **GLH-Water:** the original scene setting uses a 12,800 × 12,800 image layout.
- **UAVid and GF2-GID-15:** latency is measured in the corresponding original-image or full-scene inference setting.

The numerical results document the completed experiments; they are not a promise that a different CUDA environment, dataset preparation, training seed, checkpoint, or distributed topology will reproduce the same values exactly.

### What the modules contribute

The visual analyses associated with the experiments show the most evident improvements in the following cases:

- **GF2-GID-15:** fragmented roads, parcel boundaries, small objects, and class-transition edges, where PDE-Net reduces omissions and boundary misclassifications.
- **GLH-Water:** continuous water boundaries, narrow river channels, and shoreline transition regions, where PDE-Net suppresses fragmented omissions and restores more complete water extents.
- **UAVid:** spatially coherent urban-scene segmentation through the combined SIN and HRDE decoder refinement path.

## Installation

Create a Python environment compatible with your CUDA driver. Install a matching PyTorch and torchvision build **before** installing the remaining dependencies:

```bash
# Install PyTorch and torchvision for your CUDA environment first.
pip install -r requirements.txt
```

This project contains a local MMSegmentation-derived source tree. Run commands from the repository root, or set `PYTHONPATH=.` as shown below.

## Data preparation and checkpoints

Datasets, model weights, benchmark outputs, and manuscripts are intentionally excluded from this repository. Obtain every dataset under its own license, prepare it locally, and adapt the data roots in the corresponding configuration.

The representative GF2-GID-15 and UAVid configurations released here do not reference private `work_dirs/` checkpoints and can be trained from scratch. If you independently obtain a compatible initialization checkpoint, specify it explicitly:

```bash
PYTHONPATH=. python tools/train.py CONFIG.py \
  --cfg-options load_from=/absolute/path/to/initialization.pth
```

To evaluate a model, pass the evaluation checkpoint directly to `tools/test.py` or `tools/dist_test.sh`.

## Representative configurations

| Dataset | Configuration | Main components |
| --- | --- | --- |
| GF2-GID-15 | [`configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py`](configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py) | Spatial-parallel Swin, sinusoidal CPE, HRDE |
| UAVid | [`configs/swin/uavid/swin-large-uavid-1024x1024_sp_sin_hrde_v4_full_from512.py`](configs/swin/uavid/swin-large-uavid-1024x1024_sp_sin_hrde_v4_full_from512.py) | Spatial-parallel Swin, SIN, HRDE V4 |
| GLH-Water | [`configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_12800.py`](configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_12800.py) | Whole-scene water evaluation layout |

Before running an experiment, review its `data_root`, batch size, worker count, input size, and distributed topology. The included paths are examples from the experiment environment, not distributed data.

## Training

### Single-process training

```bash
PYTHONPATH=. python tools/train.py \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  --work-dir work_dirs/gf2_pdenet
```

### Distributed training

The launcher accepts `CONFIG GPUS [extra train arguments]`:

```bash
bash tools/dist_train.sh \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  4 \
  --work-dir work_dirs/gf2_pdenet
```

For multi-node runs, configure the following environment variables before launching:

```bash
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=your-master-host
export PORT=29500
```

Then call `tools/dist_train.sh` on each node with the appropriate `NODE_RANK`.

## Evaluation

The distributed test launcher accepts `CONFIG CHECKPOINT GPUS [extra test arguments]`:

```bash
bash tools/dist_test.sh \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  /absolute/path/to/checkpoint.pth \
  4 \
  --work-dir work_dirs/gf2_eval
```

## Latency benchmarking

`tools/benchmark_sp_latency.py` measures synchronized distributed latency and writes a JSON manifest plus raw timing records to the output directory. For a full-scene image:

```bash
torchrun --nproc_per_node=4 tools/benchmark_sp_latency.py \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  /absolute/path/to/checkpoint.pth \
  --launcher pytorch \
  --scene-img /absolute/path/to/original_scene.png \
  --output-dir benchmark_results/gf2_latency
```

Use `--nirrgb-to-rgb` for a four-channel NIR-RGB source image when channels 1, 2, and 3 should be used as RGB. Benchmark output directories are ignored by Git.

## Repository layout

```text
configs/       Shared bases and representative experiment configurations
mmcv_custom/   Custom optimizer and data-loading extensions
mmseg/         MMSegmentation-derived segmentation implementation
tools/         Training, evaluation, and latency benchmark entry points
```

## Reproducibility notes

This is a source-only research release. The reported results require prepared, licensed datasets and the relevant training or evaluation checkpoints, none of which are redistributed here. Exact results can vary with CUDA/PyTorch/MMCV versions, GPU topology, preprocessing, random seeds, and checkpoint selection.

The release is intended to make the PDE-Net implementation, configuration logic, and distributed benchmark method inspectable and reusable while keeping data and model artifacts outside the repository.

## License and attribution

This project is released for non-commercial research under the retained [CC BY-NC 4.0](LICENSE) terms. It contains code derived from REST and OpenMMLab MMSegmentation; see [NOTICE](NOTICE) for provenance and attribution details.
