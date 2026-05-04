Proximal-Based Generative Modeling for Bayesian Inverse Problems
This repository contains the official PyTorch implementation for "Proximal-Based Generative Modeling for Bayesian Inverse Problems".
Our method bridges diffusion models and proximal optimization, offering a robust framework for solving complex Bayesian inverse problems. The codebase includes a full pipeline for high-resolution image restoration tasks (e.g., inpainting, super-resolution, deblurring) focused on datasets like FFHQ, alongside robust toy experiments verifying theoretical guarantees (Gaussian equivalence, constrained quadratic optimization).
📁 Project Structure
.  
├── main.py                     # Main entry point for running experiments via CLI  
├── requirements.txt            # Python dependencies  
├── README.md                   # This file  
├── src/                        # Core implementation  
│   ├── experiment.py           # Experiment runner and evaluation pipeline  
│   ├── inverse_problems.py     # Forward operators (Inpainting, Super-Res, etc.)  
│   ├── models.py               # Enhanced UNet architectures with conditioning  
│   ├── sampling.py             # Samplers (Euler, Heun, RK4, DDIM, DiffPIR)  
│   ├── sdes.py                 # SDE formulations (VE, VP, MY)  
│   ├── training.py             # Trainer classes and dataset loaders  
│   └── utils.py                # Additional utilities, metrics, and visualization tools
└── toy_experiments/            # Theoretical verifications and 1D/2D examples  
├── counter_example_vp.py   # VP-SDE boundary accumulation visualization  
├── equivalence.py          # Score matching vs. proximal gradient equivalence  
├── gaussian_comparison.py  # 1D Gaussian comparisons  
└── proximal_quadra.py      # Constrained quadratic optimization
src/: Core implementation containing the Unet models, training loops, specific inverse problem formulations, specialized sampling procedures (including DiffPIR and gradient guidance), and utility metric calculators.
toy_experiments/: Contains 1D/2D toy distributions and constrained optimization scripts that empirically validate the equivalence of score matching and proximal gradients.
main.py: The primary entry point for launching image inverse problem experiments.
🚀 Setup & Installation
Clone the repository and navigate to the directory.
Create a virtual environment (optional but recommended):  
conda create -n prox_gen python=3.10  
conda activate prox_gen
Install dependencies:  
pip install -r requirements.txt
💾 Dataset Preparation
By default, the code focuses on FFHQ as the primary high-resolution image dataset.
Download the FFHQ dataset.
Place the images inside the ./data/FFHQ/ directory.
data/  
└── FFHQ/  
├── 00000.png  
├── 00001.png  
└── ...
(Note: You can similarly set up CelebA_HQ, LSUN, or ImageNet under ./data/)
🏃 Running Image Inverse Problems (FFHQ)
Use main.py to run experiments. The script is pre-configured to default to the FFHQ dataset and the inpainting task using our custom SDE (my).
Basic Run (FFHQ Inpainting):
python main.py --dataset FFHQ --problem inpainting --sde my --batch_size 8 --epochs 100
Super-Resolution with VP-SDE:
python main.py --dataset FFHQ --problem super_resolution --sde vp --batch_size 8
Available Arguments:
--dataset: FFHQ (default), CelebA_HQ, LSUN, ImageNet, mnist, celeba
--problem: inpainting (default), super_resolution, deblurring, nonlinear, compressed_sensing
--sde: my (default), ve, vp
--batch_size, --epochs, --lr, --data_weight
🔬 Running Toy & Theoretical Experiments
To reproduce the theoretical visualizations discussed in the paper, run the scripts located in toy_experiments/:
# Figure 1: Equivalence between score and proximal gradient  
python toy_experiments/equivalence.py
# Figure: 1D Gaussian Comparison  
python toy_experiments/gaussian_comparison.py
# Constrained Quadratic Optimization  
python toy_experiments/proximal_quadra.py
