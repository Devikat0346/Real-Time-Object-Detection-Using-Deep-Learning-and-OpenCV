import os
import random
import math
from PIL import Image

def merge_images(num_images, model_select):
    input_folder = f"viz_output_yolov8{model_select}/"
    output_folder = "merged_images/"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Select random images
    selected_images = random.sample(image_files, num_images)

    # Load images
    images = [Image.open(input_folder + img_file) for img_file in selected_images]

    # Determine the dimensions of the merged image
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    # Calculate the number of rows and columns
    grid_size = math.ceil(math.sqrt(num_images))
    total_width = grid_size * max_width
    total_height = grid_size * max_height

    # Create a new blank image with the calculated dimensions
    merged_image = Image.new('RGB', (total_width, total_height))

    # Paste the loaded images into the merged image
    x_offset = 0
    y_offset = 0
    for idx, img in enumerate(images):
        merged_image.paste(img, (x_offset, y_offset))
        x_offset += max_width

        if (idx + 1) % grid_size == 0:
            x_offset = 0
            y_offset += max_height

    # Save the merged image
    output_file = f"{output_folder}merged_yolov8{model_select}_random_{num_images}.jpg"
    merged_image.save(output_file)

    print(f"Merged image saved as: {output_file}")

# Example usage
num_images = 25
model_select = "l"
merge_images(num_images, model_select)
