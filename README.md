# PGM — Proximal-based Generative Modeling

This is a clean, self-contained PyTorch implementation of the paper *"Proximal-based Generative Modeling: Algorithm, Theory, and Applications"* (Boyang Zhang, Zhiguo Wang, Ya-Feng Liu).

https://arxiv.org/pdf/2605.13278

---

## Directory layout

```
PGM/
├── README.md                       <- this file
├── requirements.txt                <- Python dependencies
├── main.py                         <- entry point
├── pgm/                            <- core library
│   ├── config.py                   <- YAML loader
│   ├── data.py                     <- load_images / list_images_in_dir
│   ├── model.py                    <- build_model + Tweedie x0
│   ├── schedule.py                 <- build_schedule, lambda_to_t, k_us_seq / make_k_seq
│   ├── operators.py                <- GaussianDeblur / BicubicSR / RandomInpaint (forward + adjoint)
│   ├── samplers.py                 <- sample_partial (APL), sample_joint (Prox_{f+g} Langevin)
│   └── metrics.py                  <- PSNR / SSIM / LPIPS
├── configs/
│   ├── partial_deblur.yaml
│   ├── partial_sr.yaml
│   ├── partial_inpaint.yaml
│   ├── joint_deblur.yaml
│   ├── joint_sr.yaml
│   └── joint_inpaint.yaml
├── model_zoo/                      <- put the two .pt files here (see model_zoo/README.md)
│   └── README.md
├── testsets/ffhq/
│   ├── 60000.png
│   └── 60002.png
├── guided_diffusion/               <- required for model construction
└── utils/
    └── utils_model.py              <- minimal argparser (only the part used by PGM)
```

---

## 1. Install

```bash
# (recommended) create a fresh conda env
conda create -n pgm python=3.10
conda activate pgm

# clone this repo, then:
cd PGM
pip install -r requirements.txt
```

`lpips` will download the VGG weights (~140 MB) on first use.

## 2. Prepare the model weights

The two FFHQ-10M checkpoints go into `model_zoo/`:

```bash
# download from the shared Google Drive folder (see model_zoo/README.md)
# place the files at:
PGM/model_zoo/prox_ffhq_10m.pt
PGM/model_zoo/diffusion_ffhq_10m.pt
```

If your checkpoints live elsewhere, pass `--model-zoo /path/to/zoo` to
`main.py`.

## 3. Run

The default image directory is `testsets/ffhq/` (two pre-shipped FFHQ
images, `60000.png` and `60002.png`). Override with `--imgs /path/to/dir`
or `--imgs /path/to/single.png`.

```bash
# --- Partial Approximation (prox network, single-point Dirac init) ---
python main.py --config configs/partial_deblur.yaml
python main.py --config configs/partial_sr.yaml
python main.py --config configs/partial_inpaint.yaml

# --- Joint Approximation (prox network, standard-Gaussian init) ---
python main.py --config configs/joint_deblur.yaml
python main.py --config configs/joint_sr.yaml
python main.py --config configs/joint_inpaint.yaml

```

Each run saves:

```
results/<config-name>/
├── config.json               <- the resolved config
├── metrics.json              <- per-image + mean PSNR-Y / SSIM / LPIPS
├── 60000_x.npy               <- reconstruction in [-1, 1] (3x256x256)
├── 60000_y.npy               <- observation in [-1, 1]
├── 60000_mask.npy            <- only for inpaint
├── 60000.png                 <- side-by-side figure (GT | y | [mask] | recon)
├── 60002_x.npy
├── 60002_y.npy
├── 60002.png
└── ...
```

The Python routine prints one line per image, e.g.:

```
[60000] psnr=28.74  psnr_y=29.18  ssim=0.9013  lpips=0.2741  (t=[25,16,12,8], K=[25,...], 13.0s)
```

Reference numbers (single image `60000.png`, the shipped configs, 42 fixed
seed, **NFE = 100 for every sampler**).  Three design insights validated by tuning:

1. **The prox network is most accurate at small t**, so the partial deblur/SR
   schedules concentrate on t ≲ 25 with a small Langevin step;
2. **γ annealing** — `gamma_seq[i] = γ · r^(T-1-i)` (large t small γ, small t
   large γ) — is decisive whenever the schedule spans a wide t range: the chain
   *generates* content at large t (tiny γ so the data gradient does not fight the
   prior) and *refines* at small t (large γ);
3. **final-prox timestep decoupling** — the last state carries Langevin noise
   matching t ≈ 8--16, so the final `Prox_g` projection is taken at
   `final_t ∈ {8,12,16}` (projecting at the smallest t does almost nothing).


## 4. Tune the sampling parameters

All hyper-parameters live in the YAML configs. The most useful knobs:

| Key | Effect |
|---|---|
| `model_name`        | `prox_ffhq_10m` (mode, flagship) or `diffusion_ffhq_10m` (mean; use the SAME config to see the network difference) |
| `t_seq`, `K_seq`    | Explicit diffusion-timestep schedule and per-stage inner iterations (`K_seq` sums to the NFE budget) |
| `lam_max`, `lam_min`, `T` | Alternative: geometric lambda schedule (`t_seq` overrides it) |
| `gamma`, `gamma_seq` | Likelihood weight. `gamma_seq` = per-stage list for **annealing** `γ_i = γ·r^(T-1-i)` (large t small γ, small t large γ) — decisive for wide-t schedules |
| `grad_ref`          | Reference scale of the gradient `grad_f = A^T(Ax - y) / grad_ref^2`; the paper uses σ_n, we default to 0.1 to avoid the 1/σ_n² = 400 blow-up |
| `delta_scale`       | **Key knob for the prox network**: d = δ_scale·λ.  A small value (0.05--0.3) keeps the Langevin noise from washing out the mode (δ_scale=1 washes the mode out) |
| `final_prox`, `final_t` | `true` → output `Prox_g(x)` of the last state; `final_t` decouples the projection timestep (8--16 matches the final state's noise, +1--3 dB) |
| `init`              | partial: `backproj` (single-point Dirac); joint: `gauss` (standard Gaussian) |
| `clamp_x`           | safety clamp on the chain state (prevents NaN blow-ups) |

## 5. Code structure in one paragraph

`build_model` (model.py) loads the FFHQ-10M U-Net and the diffusion schedule
via `guided_diffusion.create_model_and_diffusion`. The forward operator
(operators.py) is a numpy FFT circular convolution for deblur (CPU
double-precision, robust to the GPU cuFFT race we hit on some machines) and
plain torchvision ops for SR / inpaint; it exposes only `A` and `A^T`. The two
samplers (samplers.py) implement the paper's **Annealed Proximal Langevin**
iteration in image space:

## 6. Citation

If you use this code, please cite the paper:

```bibtex
@article{zhang2026proximal,
  title={Proximal-Based Generative Modeling for Bayesian Inverse Problems},
  author={Zhang, Boyang and Wang, Zhiguo and Liu, Ya-Feng},
  journal={arXiv preprint arXiv:2605.13278},
  year={2026}
}
```


