"""
Main entry point for running inverse problem experiments.
Provides a CLI interface to configure datasets, problems, and SDE types.
"""

import argparse
import torch
import os
from src.experiment import InverseProblemExperiment

def parse_args():
    parser = argparse.ArgumentParser(description="Proximal-Based Generative Modeling for Inverse Problems")
    
    # Dataset and Problem configs
    parser.add_argument('--dataset', type=str, default='FFHQ', 
                        choices=['FFHQ', 'CelebA_HQ', 'LSUN', 'ImageNet', 'mnist', 'celeba'],
                        help="Dataset to use for the experiment")
    parser.add_argument('--problem', type=str, default='inpainting',
                        choices=['inpainting', 'super_resolution', 'deblurring', 'nonlinear', 'compressed_sensing'],
                        help="Type of inverse problem")
    parser.add_argument('--image_size', type=int, default=256, 
                        help="Image resolution (e.g., 256 for 256x256)")
    
    # Method configs
    parser.add_argument('--method', type=str, default='proximal',
                        choices=['proximal', 'diffusion', 'tv', 'dip'],
                        help="Method to use for reconstruction")
    parser.add_argument('--sde', type=str, default='my', 
                        choices=['ve', 'vp', 'my'],
                        help="Type of SDE to use")
    
    # Training configs
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--data_weight', type=float, default=10.0, help="Weight for data consistency")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"Running Inverse Problem Experiment: {args.dataset} - {args.problem}")
    print(f"Method: {args.method} | SDE: {args.sde.upper()} | Device: {device}")
    print("=" * 60)
    
    # Create required directories
    os.makedirs(f'./data/{args.dataset}', exist_ok=True)
    
    # Initialize Experiment
    experiment = InverseProblemExperiment(
        dataset_name=args.dataset,
        problem_type=args.problem,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        device=device
    )
    
    experiment_name = f"{args.method}_{args.sde}_{args.problem}_{args.dataset}"
    
    # Run the experiment
    results = experiment.run_experiment(
        method=args.method,
        sde_type=args.sde,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        data_weight=args.data_weight,
        experiment_name=experiment_name
    )
    
    if 'reconstruction_metrics' in results:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Test PSNR: {results['reconstruction_metrics']['psnr'][0]:.2f} dB")
        print(f"Test SSIM: {results['reconstruction_metrics']['ssim'][0]:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    main()