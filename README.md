This repository contains source code for reproducing the results of domain guided fine-tuning (DogFit), an efficient guidance method for transfer learning of diffusion models. 

The experimented models are SiT/XL-2 and DiT/XL-2.

We provide a demo for fine-tuning to Food-101 target dataset, as well as calculating the FID, FD_DINOV2, and Precision and Recall values.

We assume that you have conda installed on your system. To work with pure python environments, please change the create_environment() function in config.sh. 

The script creates a conda environment, downloads and preprocesses the food-101 dataset, trains the model, saves the checkpoints, generates 10,000 samples, and performs evaluations, saving all the logs and final results in .log file. 

We further provide a code for running the baselines, normal fine-tuning, CFG, DoG, and MG. 


Results on Food:

DiT with Control:
MODEL_NAME="SiT-XL/2"  # or "DiT-XL/2"
FOCUS_METRIC="FD_DINOV2"  # or "FID"

W=1
fd: 459.587829108471 
precision: 0.4735 
recall: 0.6207 
density: 0.20598000000000002 
coverage: 0.2555 
-
fd: 12.975659501012219 
precision: 0.8264 
recall: 0.5198 
density: 1.3724800000000001 
coverage: 0.9291 

w=1.5
fd: 302.8373345196511 
precision: 0.586 
recall: 0.635 
density: 0.32244000000000006 
coverage: 0.3983 
-
fd: 10.93560115737121 
precision: 0.8802 
recall: 0.4878 
density: 1.67918 
coverage: 0.9586 

w=2
fd: 228.32099029285536 
precision: 0.6655 
recall: 0.6209 
density: 0.43979999999999997 
coverage: 0.5011 
-
fd: 13.051355882080786 
precision: 0.9008 
recall: 0.4377 
density: 1.7896 
coverage: 0.9524 

w=3
fd: 199.90636520579284 
precision: 0.7441 
recall: 0.5445 
density: 0.57524 
coverage: 0.5631 
-
fd: 19.812743082584866 
precision: 0.8971 
recall: 0.3433 
density: 1.61484 
coverage: 0.8862 

w=4
fd: 219.12874392745283 
precision: 0.75 
recall: 0.4904 
density: 0.5883600000000001 
coverage: 0.5506 
-
fd: 24.83750096501344 
precision: 0.8789 
recall: 0.2845 
density: 1.4062200000000002 
coverage: 0.824 

w=5
fd: 238.86912984376175 
precision: 0.745 
recall: 0.4632 
density: 0.5729000000000001 
coverage: 0.533 
-
fd: 27.946072503246253 
precision: 0.8625 
recall: 0.2594 
density: 1.2588400000000002 
coverage: 0.7814 


To run:
  bash scripts/$SCRIPT \
    --dataset "$DATASET" \
    --server "$SERVER" \
    --cuda_devices "$CUDA_DEVICES" \
    --experiment_prename "$EXPERIMENT_PRENAME" \
    --latestart "$LATESTART" \
    --mghigh "$MGHIGH" \
    --model_name "$MODEL_NAME" \
    --guidance_control "1" \
    --w_max "$W_MAX" \
    --w_min "$W_MIN" \
    --sample_guidance "$SAMPLE_GUIDANCE" \
    --control_distribution "$CONTROL_DISTRIBUTION"

This code runs all steps, from environment creation to dataset preparation, model training and testing, and evaluation. If you wish to do any of them differently, or only do sampling, comment out sections you don't want to run in run_DogFit.sh

e.g.:


example script provided in DogFit_DiT_SiT_noControl.sh