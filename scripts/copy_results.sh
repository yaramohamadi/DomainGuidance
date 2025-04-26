#!/bin/bash

# Define source and destination directories
SRC_DIR="/projets/Ymohammadi/DomainGuidance/results"
DEST_DIR="/export/livia/home/vision/Ymohammadi/DoG/results/"

# Check if source exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Source directory '$SRC_DIR' does not exist. Exiting."
    exit 1
fi

# Create destination directory if it does not exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory '$DEST_DIR' does not exist. Creating it..."
    mkdir -p "$DEST_DIR"
fi

# Copy everything from source to destination
echo "Copying files from '$SRC_DIR' to '$DEST_DIR'..."
cp -r "$SRC_DIR"/* "$DEST_DIR"/

echo "Copy complete!"