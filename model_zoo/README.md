# Pre-trained FFHQ-10M checkpoints (256x256)

Place the two `.pt` files in this directory.

| File | What it is | Download |
|---|---|---|
| `prox_ffhq_10m.pt`           | U-Net trained by **proximal matching** (Tweedie = posterior mode / MAP) | [Google Drive folder] (https://drive.google.com/drive/folders/1mvG84ntuYa9KYMlkGhqMtkSy1syXkVmv?usp=sharing) |
| `diffusion_ffhq_10m.pt`      | U-Net trained by **denoising score matching** (Tweedie = posterior mean) |  [Google Drive folder](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) |

Both checkpoints share the same architecture (128 base channels, attention at 16x16,
learn_sigma, linear noise schedule, 1000 timesteps) and differ only in the training
objective. Both are epsilon-prediction networks, so the same Tweedie path is used
to recover x0; the qualitative difference (mode vs mean) is learned by the network
weight itself, not by the sampler.

Override the directory with `--model-zoo /path/to/your/zoo` when running `main.py`.

If you only have one checkpoint, the other will not be available -- the corresponding
configs will fail at load time. The `model_name` field in each YAML picks which
checkpoint is used.
