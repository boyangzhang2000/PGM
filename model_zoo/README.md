# Model Zoo

This directory holds pretrained image priors and the measurement-network
weights used for nonlinear deblurring. Sampling presets use the FFHQ proximal
checkpoint; training presets select the corresponding diffusion backbone as
their initialization.

## Checkpoints

| File | Purpose | Download |
| --- | --- | --- |
| `prox_ffhq_10m.pt` | FFHQ proximal approximation used by the reconstruction presets | [Google Drive](https://drive.google.com/drive/folders/1mvG84ntuYa9KYMlkGhqMtkSy1syXkVmv?usp=sharing) |
| `diffusion_ffhq_10m.pt` | FFHQ diffusion backbone for proximal training | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) |
| `imagenet256.pt` | Unconditional ImageNet diffusion backbone for proximal training | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) |
| `GOPRO_wVAE.pth` | KernelWizard measurement network for nonlinear deblurring | [Google Drive](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view) |

Place the files required by your workflow at the configured paths:

```text
model_zoo/
├── README.md
├── prox_ffhq_10m.pt
├── diffusion_ffhq_10m.pt
├── imagenet256.pt
└── GOPRO_wVAE.pth
```

The diffusion training checkpoints are not required to run the supplied FFHQ
reconstruction presets. KernelWizard weights are needed only for nonlinear
deblurring. Model weights and datasets retain their respective usage terms.

## Sampling Weights and Architectures

Set `model.checkpoint` in the sampling YAML or pass an explicit file path:

```bash
python main.py --task deblur --method partial --checkpoint /path/to/proximal.pt
```

Commands in this guide are run from the repository root. Relative resource
paths in YAML are also resolved from that root.

| Checkpoint family | Architecture definition |
| --- | --- |
| FFHQ proximal and diffusion weights | [FFHQ architecture](../configs/models/prox_ffhq.yaml) |
| ImageNet weights | [ImageNet architecture](../configs/models/prox_imagenet.yaml) |
| Exports from the training package | Architecture embedded in the checkpoint |

Legacy checkpoints contain a plain state dictionary. The loader uses the FFHQ
architecture unless `model.architecture` or `--architecture` is supplied.
For a legacy ImageNet proximal checkpoint, select its architecture explicitly:

```bash
python main.py --task deblur --method partial --checkpoint /path/to/imagenet_proximal.pt --architecture configs/models/prox_imagenet.yaml
```

Training exports load their embedded architecture automatically. An explicit
override must agree with that architecture. All weights are matched strictly
to the network; FFHQ and ImageNet definitions are not interchangeable.

Both backbones use noise prediction. Training and sampling share the conversion
to a clean-image estimate, while the learned weights determine the resulting
prior. The proximal network is treated as a learned approximation.

## Initial Weights for Training

The [FFHQ preset](../configs/train/ffhq.yaml) selects the FFHQ diffusion
checkpoint. The [ImageNet preset](../configs/train/imagenet.yaml) selects the
unconditional ImageNet checkpoint. Set `model.checkpoint` in a training
configuration or override it on the command line:

```bash
python -m train --config configs/train/ffhq.yaml --checkpoint /path/to/diffusion_ffhq_10m.pt --data /path/to/ffhq
python -m train --config configs/train/imagenet.yaml --checkpoint /path/to/imagenet256.pt --data /path/to/imagenet/train
```

An empty checkpoint setting initializes the network randomly. To continue an
existing training run, use `--resume /path/to/run/latest.pt`; a weights-only
initialization does not restore optimizer state or training progress.

The training package exports `proximal.pt` for sampling and `latest.pt` for
resumption. When validation is enabled, it also saves `best.pt`. Exported
sampling checkpoints can remain in the training output directory and be
selected with `--checkpoint`.

## Nonlinear Deblurring Assets

The [nonlinear blur configuration](../configs/operators/ndb.yaml) selects
KernelWizard weights through `KernelWizard.pretrained`. The implementation
is included in `pgm/backends/kernel_wizard/`.

For assets stored outside the repository, use `operator.engine_dir`,
`--nl-engine`, or `PGM_NDB_ENGINE`. Relative KernelWizard weight paths are
resolved from the selected asset directory. Update the operator configuration
to match that directory's layout.

## LPIPS Weights

LPIPS manages its perceptual weights through its own cache rather than this
directory. Prepare the cache before offline execution. Disabling the LPIPS
metric does not disable an LPIPS regularizer selected in the refinement
configuration.

See the [main README](../README.md) for reconstruction, training, and evaluation
workflows.
