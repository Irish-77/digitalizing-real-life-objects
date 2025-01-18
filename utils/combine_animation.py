import os
import math
import string
from PIL import Image, ImageDraw, ImageFont, ImageSequence

def scale_image_preserve_aspect(image, target_size):
    """Scale image to target size while preserving aspect ratio"""
    width, height = image.size
    aspect = width / height
    
    if aspect > 1:  # width > height
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:  # height >= width
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def create_survey_gifs(
    input_images_dir,         # Directory containing the original images
    models_parent_dir,        # Directory containing subfolders for each model
    output_gifs_dir,          # Directory to save the final combined GIFs
    label_font_path=None,     # Path to a TTF font file (optional for labeling)
    label_font_size=36,       # Increased from 20 to 36
    frame_duration=100        # Frame duration (in ms) for the final GIF
):
    """
    For each image in 'input_images_dir', combine it (static) with the side-by-side
    reconstruction GIFs found in model subfolders under 'models_parent_dir'.
    Save the combined GIF to 'output_gifs_dir'.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_gifs_dir, exist_ok=True)

    # Get the list of available models by looking at subdirectories in models_parent_dir
    model_names = [
        d for d in os.listdir(models_parent_dir)
        if os.path.isdir(os.path.join(models_parent_dir, d))
    ]
    
    # Sort model names to have a consistent ordering (optional)
    model_names.sort()

    # Prepare a sequence of label characters (A, B, C, ...)
    label_chars = list(string.ascii_uppercase)  # ['A', 'B', 'C', ...]

    # Create a single mapping file for all images
    def save_model_mapping(output_path, mappings):
        with open(output_path, 'w') as f:
            for letter, model in mappings:
                f.write(f"{letter}: {model}\n")

    # Save the mapping file at the start
    mapping_path = os.path.join(output_gifs_dir, "model_mapping.txt")
    mappings = [
        (label_chars[idx], model_name) 
        for idx, model_name in enumerate(model_names)
        if idx < len(label_chars)
    ]
    save_model_mapping(mapping_path, mappings)
    print(f"Saved model mapping: {mapping_path}")

    # Load a font if a path is provided; otherwise create a larger default font
    if label_font_path is not None and os.path.isfile(label_font_path):
        label_font = ImageFont.truetype(label_font_path, label_font_size)
    else:
        try:
            # Try to use a system font that's commonly available
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", label_font_size)
        except:
            # If that fails, create a larger default font
            default_font = ImageFont.load_default()
            label_font = default_font.font_variant(size=label_font_size)

    # Get sorted list of input images
    input_images = sorted([
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])

    # --- Process each original image ---
    for img_name in input_images:
        # Check if it’s an actual image file (you might refine checks with file extension checks)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Construct the full path to the original image
        original_img_path = os.path.join(input_images_dir, img_name)

        # Open and scale the original image to 400x400
        original_image = Image.open(original_img_path).convert("RGBA")
        original_image = scale_image_preserve_aspect(original_image, (400, 400))

        # Create white background for original image
        bg = Image.new("RGBA", (400, 400), (255, 255, 255, 255))
        # Center the scaled image on the background
        offset = ((400 - original_image.size[0]) // 2,
                 (400 - original_image.size[1]) // 2)
        bg.paste(original_image, offset, original_image)
        original_image = bg

        # Collect reconstruction GIFs (one per model)
        reconstruction_gifs = []
        for model_idx, model_name in enumerate(model_names):
            model_dir = os.path.join(models_parent_dir, model_name)
            
            # Look for a GIF that matches the input image name
            base_name = os.path.splitext(img_name)[0]
            matching_gifs = [
                f for f in os.listdir(model_dir)
                if f.lower().endswith('.gif') and base_name.lower() in f.lower()
            ]
            
            gif_path = None
            if matching_gifs:
                gif_path = os.path.join(model_dir, matching_gifs[0])
                print(f"Found matching GIF: {matching_gifs[0]} for image: {img_name}")
            else:
                print(f"No matching GIF found in {model_dir} for image: {img_name}")

            if gif_path is not None:
                # Open the GIF
                try:
                    gif_img = Image.open(gif_path)
                    # Scale each frame of the GIF to 400x240 (new dimensions)
                    frames = []
                    for frame in ImageSequence.Iterator(gif_img):
                        frame = frame.convert("RGBA")
                        scaled_frame = scale_image_preserve_aspect(frame, (400, 240))
                        # Create white background
                        bg = Image.new("RGBA", (400, 240), (255, 255, 255, 255))
                        # Center the scaled frame
                        offset = ((400 - scaled_frame.size[0]) // 2,
                                (240 - scaled_frame.size[1]) // 2)
                        bg.paste(scaled_frame, offset, scaled_frame)
                        frames.append(bg)
                    gif_img.seek(0)  # Reset gif to first frame
                    gif_img.info['frames'] = frames  # Store frames for later use
                    reconstruction_gifs.append(gif_img)
                except Exception as e:
                    print(f"Failed to open GIF at {gif_path}: {e}")
                    reconstruction_gifs.append(None)
            else:
                reconstruction_gifs.append(None)

        # If no valid GIFs found, skip
        if all(g is None for g in reconstruction_gifs):
            print(f"No GIFs found for image {img_name}. Skipping...")
            continue

        # -- Determine final canvas size --
        # New layout: 3 columns, multiple rows if needed
        gifs_per_row = 3  # Changed from 2 to 3
        num_rows = math.ceil(len(reconstruction_gifs) / gifs_per_row)
        
        # Calculate basic sizes
        gif_width = 400  # Reduced from 500 to 400 to fit three in a row
        gif_height = 240  # Reduced proportionally from 300 to 240
        horizontal_padding = 30  # Slightly reduced padding to accommodate three GIFs
        vertical_padding = 40
        bottom_padding = 80    # Extra padding for bottom
        label_height = label_font_size + 20
        
        # Get original image dimensions
        orig_w, orig_h = original_image.size
        
        # Calculate total width (3 GIFs + padding)
        total_gifs_width = (gif_width * gifs_per_row) + (horizontal_padding * (gifs_per_row + 1))
        
        # Final width should accommodate both the original image and the GIF pairs
        final_width = max(total_gifs_width, orig_w + 2 * horizontal_padding)
        
        # Calculate height for all rows of GIFs plus original image
        final_height = (
            orig_h +                                    # original image height
            vertical_padding +                         # reduced to single padding after original
            label_height +                             # height for labels
            (gif_height * num_rows) +                  # height for all GIF rows
            (vertical_padding * (num_rows + 1)) +      # padding between rows and at bottom
            bottom_padding                             # Extra padding at the bottom
        )

        # --- Build frames for the final GIF ---
        # 1) We find the max number of frames among all reconstruction GIFs.
        max_frames = 1  # we want at least 1 frame (in case of static images)
        for gif in reconstruction_gifs:
            if gif is not None:
                try:
                    frames_count = sum(1 for _ in ImageSequence.Iterator(gif))
                    max_frames = max(max_frames, frames_count)
                except:
                    pass  # if error, treat as 1 frame

        # We'll store each final frame in a list
        final_frames = []

        # For each frame index in [0, max_frames)
        for frame_idx in range(max_frames):
            # Create a new blank image (RGBA) for this frame
            frame_img = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(frame_img)

            # --- Place the original image (static) at the top center ---
            # Calculate x offset to center the original image
            orig_x = (final_width - orig_w) // 2
            orig_y = 0  # removed vertical padding from top
            frame_img.paste(original_image, (orig_x, orig_y), original_image)

            # --- Place the reconstruction GIF frames below ---
            for model_idx, gif in enumerate(reconstruction_gifs):
                # Calculate row and column position
                row = model_idx // gifs_per_row
                col = model_idx % gifs_per_row
                
                # Calculate x and y offsets for this GIF
                x_offset = horizontal_padding + (col * (gif_width + horizontal_padding))
                y_offset = (orig_y + orig_h + vertical_padding + 
                          row * (gif_height + label_height + vertical_padding))

                # Label above the GIF
                label_text = label_chars[model_idx] if model_idx < len(label_chars) else "?"
                
                # Center label above its GIF
                bbox = draw.textbbox((0, 0), label_text, font=label_font)
                label_text_width = bbox[2] - bbox[0]
                label_x = x_offset + (gif_width - label_text_width) // 2
                label_y = y_offset
                draw.text((label_x, label_y), label_text, fill=(0, 0, 0), font=label_font)

                # Place the GIF frame
                gif_frame_y = y_offset + label_height

                if gif is not None:
                    try:
                        gif.seek(frame_idx)
                        current_gif_frame = gif.info['frames'][frame_idx]
                        frame_img.paste(current_gif_frame, (x_offset, gif_frame_y), current_gif_frame)
                    except (EOFError, IndexError):
                        if gif.info['frames']:
                            current_gif_frame = gif.info['frames'][-1]
                            frame_img.paste(current_gif_frame, (x_offset, gif_frame_y), current_gif_frame)

            # Append the constructed frame to final_frames
            final_frames.append(frame_img)

        # --- Save the final GIF ---
        # Convert frames to "P" or "RGB" mode if needed and build a single GIF
        out_gif_path = os.path.join(output_gifs_dir, f"{os.path.splitext(img_name)[0]}_survey.gif")

        # PIL requires the first frame to call save with parameters, then subsequent frames appended
        first_frame = final_frames[0].convert("RGBA")
        subsequent_frames = [frm.convert("RGBA") for frm in final_frames[1:]]

        first_frame.save(
            out_gif_path,
            save_all=True,
            append_images=subsequent_frames,
            format="GIF",
            loop=0,              # loop forever
            duration=frame_duration
        )

        print(f"Saved combined survey GIF: {out_gif_path}")

        # Remove the per-image mapping file creation from the image processing loop
        # (delete or comment out the mapping file creation code at the end of the loop)


if __name__ == "__main__":
    """
    Example usage.
    Adjust the paths below to match your folder structure.
    """
    input_images_dir = "data/segmented"
    models_parent_dir = "final/"   # Each subfolder is a model
    output_gifs_dir   = "survey/animations"

    # Optional: specify a path to a TTF font if you have a specific font you want to use
    # label_font_path = "path/to/font.ttf"
    label_font_path = None

    create_survey_gifs(
        input_images_dir=input_images_dir,
        models_parent_dir=models_parent_dir,
        output_gifs_dir=output_gifs_dir,
        label_font_path=label_font_path,  # or None
        label_font_size=48,    # More reasonable large font size
        frame_duration=100
    )
