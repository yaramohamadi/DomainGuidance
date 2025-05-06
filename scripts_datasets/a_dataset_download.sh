#!/bin/bash


############################# FOOD-101 DATASET #############################
# Define target directory
TARGET_DIR="/export/datasets/public/diffusion_datasets/food_101"
# 
# # Create the directory if it does not exist
# mkdir -p "$TARGET_DIR"
# 
# # Download the dataset to the target directory
# curl -L -o "$TARGET_DIR/food-101.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101
# 
# # Unzip the dataset in the same directory
# unzip -o "$TARGET_DIR/food-101.zip" -d "$TARGET_DIR"
zip -r "$TARGET_DIR/food-101_processed.zip" "$TARGET_DIR/food-101_processed"


############################### DANISH FUNGI DATASET #############################
#TARGET_DIR="/export/datasets/public/diffusion_datasets/fungi"
#mkdir -p "$TARGET_DIR"

#curl -L -o "$TARGET_DIR/DF20-300px.tar.gz" \
#  http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz
#tar -xzf "$TARGET_DIR/DF20-300px.tar.gz" -C "$TARGET_DIR"

#curl -L -o "$TARGET_DIR/DF20-metadata.zip" \
#  http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-metadata.zip
#unzip -o "$TARGET_DIR/DF20-metadata.zip" -d "$TARGET_DIR"


################################# CALTECH-101 DATASET #############################

# TARGET_DIR="/export/datasets/public/diffusion_datasets/caltech-101"
# mkdir -p "$TARGET_DIR"
# 
# curl -L -o "$TARGET_DIR/caltech-101.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/imbikramsaha/caltech-101
# unzip -o "$TARGET_DIR/caltech-101.zip" -d "$TARGET_DIR"
# Remove the GOOGLE_BACKGROUND class
# ZIP again
# zip -r /export/datasets/public/diffusion_datasets/caltech-101/caltech-101.zip /export/datasets/public/diffusion_datasets/caltech-101/caltech-101


############################### CUB-200-2011 DATASET #############################
# TARGET_DIR="/export/datasets/public/diffusion_datasets/cub-200-2011"
# mkdir -p "$TARGET_DIR"
# 
# curl -L -o "$TARGET_DIR/cub-200-2011.zip" \
#   https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011
# unzip -o "$TARGET_DIR/cub-200-2011.zip" -d "$TARGET_DIR"

############################### Stanford Cars DATASET #############################

# TARGET_DIR="/export/datasets/public/diffusion_datasets/stanford-cars"
#mkdir -p "$TARGET_DIR"
# 
#curl -L -o "$TARGET_DIR/stanford-cars.zip" \
#  https://www.kaggle.com/api/v1/datasets/download/eduardo4jesus/stanford-cars-dataset
#unzip -o "$TARGET_DIR/stanford-cars.zip" -d "$TARGET_DIR"

# Get test annotation files
# curl -L -o $TARGET_DIR/stanford-cars_test_anno.zip\
#  https://www.kaggle.com/api/v1/datasets/download/abdelrahmant11/standford-cars-dataset-meta
# unzip -o "$TARGET_DIR/stanford-cars_test_anno.zip" -d "$TARGET_DIR"
# Run the cars_preprocess.py script to get the train split annotation folders
# zip 

# Get metadata for test dataaset too! 
curl -L -o ~/Downloads/standford-cars-dataset-meta.zip\
  https://www.kaggle.com/api/v1/datasets/download/abdelrahmant11/standford-cars-dataset-meta


pushd /export/datasets/public/diffusion_datasets/stanford-cars/ > /dev/null
zip -r stanford-cars_processed.zip stanford-cars_processed/*
popd > /dev/null
