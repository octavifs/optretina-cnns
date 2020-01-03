"""
Testing for an arbitrary number of images, already preprocessed. Binary or not.
"""


import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

VALID_IMAGE_FILE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".gif"
]

# ASCII art source: http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=SAURON%0ASCORING%0ASERVICE
TITLE = """
███████╗ █████╗ ██╗   ██╗██████╗  ██████╗ ███╗   ██╗   
██╔════╝██╔══██╗██║   ██║██╔══██╗██╔═══██╗████╗  ██║   
███████╗███████║██║   ██║██████╔╝██║   ██║██╔██╗ ██║   
╚════██║██╔══██║██║   ██║██╔══██╗██║   ██║██║╚██╗██║   
███████║██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║ ╚████║   
╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   
                                                       
███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝ 
███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                       
███████╗███████╗██████╗ ██╗   ██╗██╗ ██████╗███████╗   
██╔════╝██╔════╝██╔══██╗██║   ██║██║██╔════╝██╔════╝   
███████╗█████╗  ██████╔╝██║   ██║██║██║     █████╗     
╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██║██║     ██╔══╝     
███████║███████╗██║  ██║ ╚████╔╝ ██║╚██████╗███████╗   
╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚═╝ ╚═════╝╚══════╝   
                                                       """


def parse_cli_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return args


def add_args(parser):
    parser.add_argument(
        "images_path",
        type=str,
        help="images path"
    )
    parser.add_argument(
        "crop_size",
        type=int,
        help="crop size for the images"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="model path"
    )
    parser.add_argument(
        "csv_output",
        type=str,
        help="csv predictions output"
    )
    parser.add_argument(
        "--color",
        choices=["rgb", "grayscale"],
        default="rgb",
        help="Color mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--column",
        default="image_id",
        help="Column to read from the csv file. image_id by default"
    )


COLORMODE = {
    "rgb": "RGB",
    "grayscale": "L"
}


def batch_generator(base_image_path, image_id_list, batch_size, colormode, crop_size):
    num_batches = ceil(len(image_id_list) / batch_size)
    for num_batch in range(num_batches):
        start = num_batch * batch_size
        end = start + batch_size
        batch_images_paths = image_id_list[start:end]

        images = [
            Image.open(base_image_path / image_id)
                .convert(mode=COLORMODE[colormode])
                .resize((crop_size, crop_size))
            for image_id in batch_images_paths
        ]
        if colormode == "grayscale":
            images_arr = [np.array(img)[:, :, np.newaxis] for img in images]
        else:
            images_arr = [np.array(img) for img in images]

        # Normalize pixel values between 0 and 1
        images_test_arr = np.array(images_arr) / 255.0

        # Return prepared batch
        yield images_test_arr


def main():
    args = parse_cli_args()

    # Fancy presentation stuff
    print(TITLE)
    print("Parameters:")
    print("Model: %s" % args.model_path)
    print("Test images: %s" % args.images_path)
    print("Crop size: %d" % args.crop_size)
    print("Predictions output: %s" % args.csv_output)

    # Prepare dataframe with input images
    image_id_list = []
    base_image_path = Path(args.images_path)
    for image_extension in VALID_IMAGE_FILE_EXTENSIONS:
        images = [img.name for img in base_image_path.glob("*" + image_extension)]
        image_id_list.extend(images)
    test_df = pd.DataFrame(image_id_list, columns=[args.column])

    # Check if there are any images
    if not len(test_df):
        print(f"No images were found on {args.images_path}\nPlease check the images_path argument.")
        return

    # Prepare ImageDataGenerator
    print("\nPreparing test images...")

    num_steps = ceil(len(image_id_list) / args.batch_size)
    image_batch_generator = batch_generator(
        base_image_path,
        image_id_list,
        args.batch_size,
        args.color,
        args.crop_size
    )

    # Load the model
    print("\nLoading model...")
    model = load_model(args.model_path)

    # Perform model evaluation
    print("\nEvaluating model...")
    predictions_arr = model.predict_generator(image_batch_generator, steps=num_steps, verbose=1)
    predictions_df = pd.DataFrame(predictions_arr)

    # Get mapping of classes, if available
    class_indices_path = Path(args.model_path + ".class_indices")
    if class_indices_path.exists():
        with class_indices_path.open("r") as fd:
            class_indices_arr = json.load(fd)
            predictions_df.columns = class_indices_arr

    # Prepare predictions dataframe
    predicted_df = pd.concat([test_df, predictions_df], axis=1)
    predicted_df = predicted_df.dropna()
    predicted_df = pd.melt(predicted_df, id_vars=test_df.columns, var_name="label", value_name="score")
    predicted_df = predicted_df.sort_values([args.column, "label"])

    # Save it to csv
    predicted_df.to_csv(args.csv_output, index=False, header=True)
    print("\nDone!")


if __name__ == "__main__":
    main()
