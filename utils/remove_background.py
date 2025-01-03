#!/usr/bin/env python3

"""Script Name: remove_background.py

Description:
    This script uses Meta's Segment Anything Model (SAM) to segment the main
    object from images. It supports both single-image and multi-image (folder)
    inputs. Users can interactively specify a bounding box or points to guide the
    segmentation. The final segmentation is saved as an RGBA image (with
    transparent background for the non-object regions).

Dependencies:
    - python >= 3.7
    - torch >= 1.7.0
    - opencv-python
    - pillow
    - segment_anything (install via: pip install git+https://github.com/facebookresearch/segment-anything.git)
    - A valid SAM checkpoint file (e.g., sam_vit_h.pth), but can also be downloaded.
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional, Union, Any
import glob
import cv2
import torch
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import urllib.request
import tqdm

DEFAULT_MODEL_NAME = "sam_vit_h_4b8939.pth"
MODEL_DIR = "model"
DEFAULT_MODEL_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{DEFAULT_MODEL_NAME}"

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    print("ERROR: segment_anything package not found. "
          "Install via: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)


def download_default_model() -> Optional[str]:
    """
    Download the default SAM model if it doesn't exist.
    
    Returns:
        Optional[str]: Path to the model file or None if download failed
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
    
    if os.path.exists(model_path):
        return model_path
        
    print(f"Downloading default model from {DEFAULT_MODEL_URL}")
    try:
        with urllib.request.urlopen(DEFAULT_MODEL_URL) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(model_path, 'wb') as f, tqdm.tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as progress_bar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    progress_bar.update(len(buffer))
        
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with checkpoint, input, output, device, etc.
    """
    parser = argparse.ArgumentParser(
        description="Segment an object from an image or folder of images using Segment Anything Model (SAM)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the SAM checkpoint file (e.g., sam_vit_h.pth). "
             "If not provided, will use or download the default model."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        help="SAM model type (e.g., vit_h, vit_l, vit_b)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an input image or a folder of images."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where segmented RGBA output(s) will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Compute device to use. Options: 'cpu', 'cuda', or 'auto'. "
             "Default is 'auto', which picks CUDA if available, else CPU."
    )
    parser.add_argument(
        "--no-interaction",
        action="store_true",
        help="If set, tries to segment without interactive bounding box/point selection. "
             "Not recommended if you must precisely choose a region. Uses entire image or fails."
    )

    return parser.parse_args()


def select_bounding_box_and_points(
    image: NDArray[np.uint8]
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[List[Tuple[int, int]], List[int]]]]:
    """
    Open an OpenCV window to allow the user to optionally select a bounding box (ROI)
    and then optionally select multiple points with mouse clicks.
    
    Args:
        image: The image to display.
    
    Returns:
        Tuple containing:
            - bounding_box: (x, y, w, h) of the selected ROI or None
            - points_and_labels: Tuple of (points, labels) where points is list of (x, y) 
              coordinates and labels is list of 1 (foreground) or 0 (background)
    """
    # Print instructions in the terminal
    print("""
======================================================
Step 1: Bounding Box Selection (Optional)
------------------------------------------------------
1) A window titled "Select ROI" will appear.
2) Left-click and drag to draw a rectangular bounding
   box around the main object.
3) Press ENTER or SPACE to confirm the bounding box.
4) Press ESC to skip bounding box selection and proceed
   to point selection.
======================================================
""")

    clone = image.copy()
    
    # Step 1: Let user select bounding box (optional)
    cv2.imshow("Select ROI", clone)
    bbox = cv2.selectROI("Select ROI", clone, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    
    (x, y, w, h) = bbox
    has_bbox = w != 0 and h != 0
    
    # Print instructions for selecting points
    print("""
======================================================
Step 2: Points Selection (Optional)
------------------------------------------------------
1) A window titled "Select Points (Press ESC when done)"
   will appear.
2) Left-click to add foreground points (red dots)
3) Right-click to add background points (blue dots)
4) Press ESC (or 'q') when you're done placing points.
5) If no points are selected, the model will try to
   segment based on the bounding box (if provided) or
   the entire image.
======================================================
""")

    points = []
    point_labels = []

    def mouse_callback(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click = foreground
            cv2.circle(param, (mx, my), 3, (0, 0, 255), -1)  # Red dot
            points.append((mx, my))
            point_labels.append(1)
            cv2.imshow("Select Points (Press ESC when done)", param)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click = background
            cv2.circle(param, (mx, my), 3, (255, 0, 0), -1)  # Blue dot
            points.append((mx, my))
            point_labels.append(0)
            cv2.imshow("Select Points (Press ESC when done)", param)

    temp_img = clone.copy()
    # Draw the bounding box on the temporary image if one was selected
    if has_bbox:
        cv2.rectangle(temp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Select Points (Press ESC when done)", temp_img)
    cv2.setMouseCallback("Select Points (Press ESC when done)", mouse_callback, temp_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # Press ESC or 'q' to finish
        if key == 27 or key == ord('q'):
            break
    cv2.destroyWindow("Select Points (Press ESC when done)")

    return (bbox if has_bbox else None), (points if points else None, point_labels if points else None)


def load_image_paths(input_path: str) -> List[str]:
    """
    Given a file or directory path, return a list of image file paths.
    
    Args:
        input_path: The path to an image file or a directory of images.
    
    Returns:
        List of image file paths.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Collect common image extensions
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        return files
    else:
        print(f"ERROR: Invalid input path: {input_path}")
        return []


def initialize_sam_model(
    checkpoint_path: str,
    model_type: str,
    device: str
) -> Any:  # Using Any because SamPredictor type isn't available
    """
    Initialize the SAM model (SamPredictor) using the given checkpoint and device.
    
    Args:
        checkpoint_path: Path to the SAM checkpoint file.
        model_type: Model variant (e.g., vit_h, vit_l, vit_b).
        device: 'cpu' or 'cuda'.
    
    Returns:
        The initialized SAM predictor instance.
    """
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    return predictor


def segment_object(
    predictor: Any,
    image_bgr: NDArray[np.uint8],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    points: Optional[Tuple[List[Tuple[int, int]], List[int]]] = None,
) -> NDArray[np.uint8]:
    """
    Perform segmentation on an image using SAM, with optional bounding box and points.

    Args:
        predictor: SAM predictor object.
        image_bgr: Input image in BGR.
        bbox: (x, y, w, h) bounding box.
        points: Tuple of (point_coords, point_labels) where point_coords is list of (x, y)
               coordinates and point_labels is list of 1 (foreground) or 0 (background).

    Returns:
        Binary mask (shape: [H, W]) of the segmented object.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_boxes = None
    input_points = None
    input_labels = None

    if bbox is not None:
        x, y, w, h = bbox
        input_box = np.array([x, y, x + w, y + h])
        input_boxes = input_box[None, :]

    if points is not None:
        point_coords, point_labels = points
        if point_coords:
            input_points = np.array(point_coords)
            input_labels = np.array(point_labels)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_boxes,
        multimask_output=True
    )

    # If multiple masks exist, let user choose the best one interactively
    if masks.shape[0] > 1:
        chosen_mask = interactive_mask_selection(image_bgr, masks)
    else:
        chosen_mask = masks[0]

    # Convert mask from bool to uint8 for easier processing
    binary_mask = chosen_mask.astype(np.uint8)
    return binary_mask


def interactive_mask_selection(
    image_bgr: NDArray[np.uint8],
    masks: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """
    Show multiple candidate masks to the user and let them choose the best one
    by pressing a numeric key (1, 2, 3, ...).
    
    Args:
        image_bgr: Original BGR image.
        masks: Candidate masks of shape [N, H, W].

    Returns:
        The user-selected mask.
    """
    mask_count = masks.shape[0]
    
    # Print step instructions in the terminal
    print(f"""
======================================================
Step 3: Mask Selection - We found {mask_count} candidate masks
------------------------------------------------------
1) Each candidate mask will be shown in a separate window.
2) Press a digit key (1..{mask_count}) to select that mask.
3) Press space to move to the next mask.
4) Press ESC to choose the first mask by default.
======================================================
""")

    preview_window = "Select Mask - press a number key (1..N) or ESC to pick the first by default."

    for idx in range(mask_count):
        overlay = image_bgr.copy()
        overlay[masks[idx] == 1] = [0, 255, 0]  # highlight the mask in green
        cv2.imshow(preview_window, overlay)
        print(f"Showing mask {idx + 1}/{mask_count}, press {idx + 1} to select.")

        key = cv2.waitKey(0) & 0xFF
        # If user presses a digit corresponding to the mask index
        if key == 27:  # ESC
            print("ESC pressed. Selecting the first mask by default.")
            cv2.destroyWindow(preview_window)
            return masks[0]
        elif key >= ord('1') and key <= ord(str(mask_count)):
            chosen_index = int(chr(key)) - 1
            print(f"User selected mask {chosen_index + 1}.")
            cv2.destroyWindow(preview_window)
            return masks[chosen_index]

    # Fallback: if user never pressed an appropriate key
    print("No valid key pressed. Selecting the first mask by default.")
    cv2.destroyWindow(preview_window)
    return masks[0]


def save_segmented_image(
    image_bgr: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    output_path: str
) -> None:
    """
    Save the segmented object in RGBA format with a transparent background
    outside the mask region.

    Args:
        image_bgr: Original BGR image.
        mask: Binary mask of shape [H, W].
        output_path: Path to save the RGBA image.
    """
    # Convert to RGBA with Pillow
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb).convert("RGBA")
    np_img = np.array(pil_img)

    # mask == 1 => alpha=255 (fully opaque), else alpha=0 (transparent)
    alpha_channel = (mask * 255).astype(np.uint8)
    np_img[:, :, 3] = alpha_channel

    # Save result
    result = Image.fromarray(np_img, "RGBA")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Saved segmented RGBA image to: {output_path}")


def main() -> None:
    args = parse_arguments()

    # Handle model selection/download
    if args.checkpoint is None:
        print("No checkpoint specified, looking for default model...")
        model_path = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
        if not os.path.exists(model_path):
            model_path = download_default_model()
            if model_path is None:
                print("Failed to download default model. Please provide a checkpoint path.")
                sys.exit(1)
        args.checkpoint = model_path

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: SAM checkpoint not found at: {args.checkpoint}")
        sys.exit(1)

    # Determine device
    if args.device == "auto" or args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Initialize the SAM predictor
    try:
        predictor = initialize_sam_model(args.checkpoint, args.model_type, device)
    except Exception as e:
        print(f"ERROR initializing SAM model: {e}")
        sys.exit(1)

    # Collect input images
    image_paths = load_image_paths(args.input)
    if not image_paths:
        print("No valid images found. Exiting.")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Process each image
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\nProcessing ({idx}/{len(image_paths)}): {img_path}")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Failed to read image: {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{base_name}_segmented.png")

        if args.no_interaction:
            # Attempt to segment the entire image (no bounding box / points).
            try:
                binary_mask = segment_object(predictor, image_bgr, bbox=None, points=None)
            except Exception as e:
                print(f"Segmentation failed for {img_path} in no-interaction mode: {e}")
                continue
        else:
            # Interactive bounding box and points selection
            bbox, points_and_labels = select_bounding_box_and_points(image_bgr)
            if bbox is None and points_and_labels[0] is None:
                print("Skipping this image due to no bounding box and no points.")
                continue

            binary_mask = segment_object(predictor, image_bgr, bbox=bbox, points=points_and_labels)

        # Save the RGBA result
        save_segmented_image(image_bgr, binary_mask, out_path)


if __name__ == "__main__":
    main()
