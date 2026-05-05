# **Proximal-Based Generative Modeling for Bayesian Inverse Problems**



This repository contains the official PyTorch implementation for "Proximal-Based Generative Modeling for Bayesian Inverse Problems".



Our method bridges diffusion models and proximal optimization, offering a robust framework for solving complex Bayesian inverse problems. The codebase includes a full pipeline for high-resolution image restoration tasks (e.g., inpainting, super-resolution, deblurring) focused on datasets like FFHQ, alongside robust toy experiments verifying theoretical guarantees (Gaussian equivalence, constrained quadratic optimization).



## **📁 Project Structure**

.  
├── main.py                     # Main entry point for running experiments via CLI  
├── requirements.txt            # Python dependencies  
├── README.md                   # This file  
├── src/                        # Core implementation  
│   ├── experiment.py           # Experiment runner and evaluation pipeline  
│   ├── inverse\_problems.py     # Forward operators (Inpainting, Super-Res, etc.)  
│   ├── models.py               # Enhanced UNet architectures with conditioning  
│   ├── sampling.py             # Samplers (Euler, Heun, RK4, DDIM, DiffPIR)  
│   ├── sdes.py                 # SDE formulations (VE, VP, MY)  
│   ├── training.py             # Trainer classes and dataset loaders  
│   └── utils.py                # Additional utilities, metrics, and visualization tools  
└── toy\_experiments/            # Theoretical verifications and 1D/2D examples  
    ├── counter\_example\_vp.py   # VP-SDE boundary accumulation visualization  
    ├── equivalence.py          # Score matching vs. proximal gradient equivalence  
    ├── gaussian\_comparison.py  # 1D Gaussian comparisons  
    └── proximal\_quadra.py      # Constrained quadratic optimization

* src/: Core implementation containing the Unet models, training loops, specific inverse problem formulations, specialized sampling procedures (including DiffPIR and gradient guidance), and utility metric calculators.
* toy\_experiments/: Contains 1D/2D toy distributions and constrained optimization scripts that empirically validate the equivalence of score matching and proximal gradients.
* main.py: The primary entry point for launching image inverse problem experiments.



## **🚀 Setup \& Installation**

1. Clone the repository and navigate to the directory.
2. Create a virtual environment (optional but recommended):  
conda create -n prox\_gen python=3.10  
conda activate prox\_gen
3. Install dependencies:  
pip install -r requirements.txt



## **💾 Dataset Preparation**

By default, the code focuses on **FFHQ** as the primary high-resolution image dataset.

1. Download the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).
2. Place the images inside the ./data/FFHQ/ directory.

data/  
└── FFHQ/  
  ├── 00000.png  
  ├── 00001.png  
  └── ...

*(Note: You can similarly set up CelebA\_HQ, LSUN, or ImageNet under ./data/)*



## **🏃 Running Image Inverse Problems (FFHQ)**

Use main.py to run experiments. The script is pre-configured to default to the FFHQ dataset and the inpainting task using our custom SDE (my).

**Basic Run (FFHQ Inpainting):**

python main.py --dataset FFHQ --problem inpainting --sde my --batch\_size 8 --epochs 100

**Super-Resolution with VP-SDE:**

python main.py --dataset FFHQ --problem super\_resolution --sde vp --batch\_size 8

**Available Arguments:**

* \--dataset: FFHQ (default), CelebA\_HQ, LSUN, ImageNet, mnist, celeba
* \--problem: inpainting (default), super\_resolution, deblurring, nonlinear, compressed\_sensing
* \--sde: my (default), ve, vp
* \--batch\_size, --epochs, --lr, --data\_weight



## **🔬 Running Toy \& Theoretical Experiments**

To reproduce the theoretical visualizations discussed in the paper, run the scripts located in toy\_experiments/:

\# Figure 1: Equivalence between score and proximal gradient  
python toy\_experiments/equivalence.py

\# Figure: 1D Gaussian Comparison  
python toy\_experiments/gaussian\_comparison.py

\# Constrained Quadratic Optimization  
python toy\_experiments/proximal\_quadra.py

