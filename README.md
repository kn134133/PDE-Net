# PDE-Net: Preserving Spatial Continuity and Fine-Scale Details in End-to-End Semantic Segmentation of Large-Swath Remote Sensing Imagery

PDE-Net is a research implementation for end-to-end semantic segmentation of large-swath remote sensing imagery. It extends a spatial-parallel Swin Transformer and UPerNet pipeline with positional and detail-preserving components for distributed whole-image processing.

## Highlights

- Spatial-parallel Swin backbone for partitioned multi-GPU inference with cross-region feature interaction.
- Sinusoidal cross-region positional encoding (CPE) in `SwinTransformer_sp_layer_cpe`.
- Spatial Information Network (SIN) decoder component for spatial and channel feature refinement.
- High-Resolution Detail Enhancement (HRDE) decoder component for edge-aware, multi-scale detail refinement.
- Representative configurations for GF2-GID-15, UAVid, and GLH-Water experiments.
- Four-GPU experiments use a 2 × 2 spatial partition for end-to-end whole-image processing.

## Installation

Create a Python environment compatible with your CUDA driver, then install the matching PyTorch and torchvision builds first. Install the remaining Python packages from this repository afterwards:

```bash
pip install -r requirements.txt
```

This project contains a local MMSegmentation-derived source tree. Run commands from the repository root, or set `PYTHONPATH=.` as shown below.

## Data and model weights

Datasets, checkpoints, benchmark outputs, and manuscript files are **not distributed** with this repository. Obtain every dataset under its own license and place it in a local `data/` directory matching the relevant configuration.

The released representative configurations train from scratch and do not require a local checkpoint. To initialize an experiment from a checkpoint you obtain independently, supply it explicitly, for example:

```bash
PYTHONPATH=. python tools/train.py CONFIG.py --cfg-options load_from=/path/to/initialization.pth
```

For evaluation, provide a checkpoint explicitly to the test command.

## Representative configurations

| Dataset | Configuration | Notes |
| --- | --- | --- |
| GF2-GID-15 | `configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py` | Spatial-parallel Swin with sinusoidal CPE and HRDE. |
| UAVid | `configs/swin/uavid/swin-large-uavid-1024x1024_sp_sin_hrde_v4_full_from512.py` | Spatial-parallel Swin with SIN and HRDE V4. |
| GLH-Water | `configs/swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-sp_test_12800.py` | Whole-scene evaluation layout for large water scenes. |

Adjust each configuration’s `data_root`, batch size, worker count, and distributed topology for your environment and prepared dataset.

## Training

For a single process:

```bash
PYTHONPATH=. python tools/train.py \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  --work-dir work_dirs/gf2_pdenet
```

For distributed training, the wrapper accepts `CONFIG GPUS [extra train arguments]`:

```bash
bash tools/dist_train.sh \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  4 \
  --work-dir work_dirs/gf2_pdenet
```

For multi-node runs, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `PORT` before calling the wrapper.

## Evaluation

The distributed test wrapper accepts `CONFIG CHECKPOINT GPUS [extra test arguments]`:

```bash
bash tools/dist_test.sh \
  configs/swin/gf2/swin-large_upernet_1024x1024_gf2_sp_cpe_sin_hrde.py \
  /path/to/checkpoint.pth \
  4 \
  --work-dir work_dirs/gf2_eval
```

## Repository layout

```text
configs/       Experiment configurations and shared bases
mmcv_custom/   Custom optimizer and data-loading extensions
mmseg/         MMSegmentation-derived segmentation implementation
tools/         Training, evaluation, and benchmark entry points
```

## License and attribution

This project is released for non-commercial research under the retained [CC BY-NC 4.0](LICENSE) terms. It contains code derived from REST and OpenMMLab MMSegmentation; see [NOTICE](NOTICE) for provenance and attribution details.
