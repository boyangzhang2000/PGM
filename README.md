# PGM — Proximal-based Generative Modeling

PGM solves image inverse problems with learned proximal maps and annealed
proximal Langevin sampling. It provides **Joint** and **Partial** methods with
**EM**, **exponential interpolation**, and **linearly implicit IMEX**
discretizations, together with proximal training for FFHQ and ImageNet.

Sampling, measurement operators, and optional image refinement share a
YAML-driven interface. The runtime uses an available CUDA device and otherwise
falls back to the CPU.

## Supported Tasks

| Task | Measurement model | Partial configuration | Joint configuration |
| --- | --- | --- | --- |
| `deblur` | Gaussian blur | [Partial](configs/partial/deblur.yaml) | [Joint](configs/joint/deblur.yaml) |
| `inpaint` | Randomly missing pixels | [Partial](configs/partial/inpaint.yaml) | [Joint](configs/joint/inpaint.yaml) |
| `sr` | Antialiased downsampling | [Partial](configs/partial/sr.yaml) | [Joint](configs/joint/sr.yaml) |
| `ndb` | KernelWizard nonlinear blur | [Partial](configs/partial/ndb.yaml) | [Joint](configs/joint/ndb.yaml) |
| `pr` | Fourier-amplitude measurements | [Partial](configs/partial/pr.yaml) | [Joint](configs/joint/pr.yaml) |

## Repository Layout

```text
.
├── main.py                 Reconstruction entry point
├── requirements.txt        Python dependencies
├── configs/
│   ├── defaults.yaml       Shared sampling settings
│   ├── joint/              Joint task presets
│   ├── partial/            Partial task presets
│   ├── models/             FFHQ and ImageNet architectures
│   ├── operators/          Nonlinear operator settings
│   └── train/              Training presets
├── pgm/                    Samplers, operators, solvers, and evaluation
│   └── backends/           Nonlinear measurement networks
├── train/                  Moreau score matching and checkpoint management
├── guided_diffusion/       Diffusion network components
├── model_zoo/              Pretrained weights
└── testsets/               Reconstruction inputs
```

## Installation

Run the following commands from the repository root in your Python environment:

```bash
python -m pip install -r requirements.txt
```

GPU execution requires a CUDA-compatible PyTorch installation. LPIPS may
download its perceptual weights on first use; prepare its cache before running
offline.

## Prepare Weights and Images

Download the FFHQ proximal checkpoint and place it in `model_zoo/`.
The [model zoo guide](model_zoo/README.md) lists the checkpoint files, available
downloads, architecture choices, and additional assets required for training
and nonlinear deblurring.

The default reconstruction inputs are in `testsets/ffhq/`. Use `--imgs` to
select another image directory or a single image. The reconstruction pipeline
creates measurements from the input images and uses the clean inputs as
evaluation references.

## Image Reconstruction

Select a task and method to load their preset:

```bash
python main.py --task inpaint --method partial
python main.py --task deblur --method partial
python main.py --task sr --method partial
python main.py --task ndb --method joint
python main.py --task pr --method joint
```

Select a discretization, an explicit configuration, or custom input and output
paths as needed:

```bash
python main.py --task deblur --method partial --scheme exp
python main.py --task deblur --method joint --scheme imp
python main.py --config configs/partial/pr.yaml --device cpu
python main.py --task sr --method partial --imgs /path/to/images --out results/restoration
```

`python -m pgm` provides the same interface. Use either `--config` or the
`--task` and `--method` pair. Choose an empty output directory for each run.

| Option | Purpose |
| --- | --- |
| `--scheme` | Select `em`, `exp`, or `imp` and apply its preset overrides |
| `--device` | Select automatic device detection, CPU, or CUDA |
| `--checkpoint` | Load weights from a custom path |
| `--architecture` | Select the network definition for a legacy state dictionary |
| `--image-offset` | Preserve image indices, seeds, and masks when replaying a subset |
| `--dry-run` | Validate and print the resolved configuration without loading weights |
| `--smoke` | Run a reduced execution check |
| `--lpips` | Control perceptual metric evaluation |

`--lpips off` disables the perceptual metric. A configuration that enables
LPIPS regularization still requires the perceptual model.

### Configuration

[Shared defaults](configs/defaults.yaml) are overridden by the selected task
preset and its discretization settings. YAML files remain the source of
parameter values; see the [configuration reference](configs/README.md).

| Configuration group | Controls |
| --- | --- |
| `model` | Checkpoint, architecture, and inference precision |
| `operator` | Measurement geometry, noise, kernels, and external assets |
| `sampler` | Initial state, annealing times, inner-step allocation, step sizes, guidance, and readout |
| `consistency` | Observation constraints and optional in-chain corrections |
| `refinement` | Post-sampling optimizer and data, LPIPS, and TV terms |
| Seed fields | Independent sampling and observation streams and their per-image offsets |

The sampling budget includes an enabled final proximal readout. Post-sampling
Adam or L-BFGS refinement optimizes the image without calling or differentiating
through the proximal network. Its iterations and operator evaluations are
recorded separately from sampling network evaluations.

### Outputs

Each run records the effective configuration, input and weight digests, device,
seeds, metrics, and evaluation counts. Reconstruction outputs are organized as:

```text
results/<run>/
├── config.yaml
├── config.json
├── metrics.json
├── metrics.csv
├── source_<digest>.zip
└── <image-stem>/
    ├── reconstruction.png
    ├── comparison.png
    ├── reconstruction.npy
    ├── sampling_estimate.npy
    ├── raw_estimate.npy
    ├── observation.npy
    ├── state.npy
    └── metrics.json
```

Inpainting also saves the observation mask. The sampling estimate is retained
before independent refinement; the raw estimate precedes final observation
constraints. Metrics include PSNR, luminance PSNR, SSIM, and LPIPS when enabled.

## Training

The `train` package fine-tunes unconditional diffusion backbones using
Gaussian-kernel Moreau score matching. Training and sampling use the same
explicit time conditioning and clean-image prediction.

Prepare an FFHQ image directory or an ImageNet training directory with class
subdirectories, then select the corresponding preset:

```bash
python -m train --config configs/train/ffhq.yaml --data /path/to/ffhq
python -m train --config configs/train/imagenet.yaml --data /path/to/imagenet/train
```

Images are discovered recursively. Class directory names organize the dataset;
they are not network conditioning labels. Architecture, initial weights, kernel
annealing, optimizer, augmentation, and output settings are defined in
`configs/train/`.

Use command-line paths to select initial weights, validation data, or resume a
saved run:

```bash
python -m train --config configs/train/ffhq.yaml --checkpoint /path/to/pretrained.pt --data /path/to/ffhq
python -m train --config configs/train/imagenet.yaml --data /path/to/imagenet/train --validation /path/to/imagenet/val
python -m train --config /path/to/training.yaml --resume /path/to/run/latest.pt
```

| Training output | Purpose |
| --- | --- |
| `proximal.pt` | Current network weights with an embedded architecture for sampling |
| `latest.pt` | Model, optimizer, random state, and data cursor for resumption |
| `best.pt` | Best model under fixed-corruption validation, when validation is enabled |
| `config.yaml`, `run.json`, `history.jsonl`, `status.json` | Configuration, provenance, training statistics, and progress |

Load an exported proximal model directly for reconstruction:

```bash
python main.py --task deblur --method partial --checkpoint /path/to/run/proximal.pt
```

Relative resource paths in YAML are resolved from the repository root.
Command-line file paths are resolved from the current working directory.

## Citation
If you use this code, please cite the paper:
```bibtex
@article{zhang2026proximal,
  title={Proximal-Based Generative Modeling for Bayesian Inverse Problems},
  author={Zhang, Boyang and Wang, Zhiguo and Liu, Ya-Feng},
  journal={arXiv preprint arXiv:2605.13278},
  year={2026}
}
```

https://arxiv.org/pdf/2605.13278
