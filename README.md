This repository contains source code for reproducing the results of domain guided fine-tuning (DogFit), an efficient guidance method for transfer learning of diffusion models. 

The experimented models are SiT/XL-2 and DiT/XL-2.

We provide a demo for fine-tuning to Food-101 target dataset, as well as calculating the FID, FD_DINOV2, and Precision and Recall values.

We assume that you have conda installed on your system. To work with pure python environments, please change the create_environment() function in config.sh. 

The script creates a conda environment, downloads and preprocesses the food-101 dataset, trains the model, saves the checkpoints, generates 10,000 samples, and performs evaluations, saving all the logs and final results in .log file. 

We further provide a code for running the baselines, normal fine-tuning, CFG, DoG, and MG. 