#!/bin/bash

# Directory containing the images
IMAGE_DIR="/home/ntomasz/Documents/Personal/OBJECTS/repo/data/segmented"
SCRIPT_DIR="/home/ntomasz/Documents/Personal/OBJECTS/InstantMesh"
OUT_DIR="/home/ntomasz/Documents/Personal/OBJECTS/repo/out"
cd $SCRIPT_DIR
source /home/ntomasz/Documents/Personal/OBJECTS/envs/InstantMesh/bin/activate
# Check if directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory $IMAGE_DIR not found!"
    exit 1
fi

# Iterate over all PNG files in the directory
for image in "$IMAGE_DIR"/*.png; do
    # Check if file exists (in case no PNGs are found)
    if [ -f "$image" ]; then
        echo "Processing: $image"
        python run.py configs/instant-mesh-large.yaml "$image" --save_video --no_rembg --output_path "$OUT_DIR"
    fi
done

echo "Processing complete!"

