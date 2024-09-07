#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read the dataset
    logger.info(f"Reading data from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    # Basic data cleaning: remove outliers based on min_price and max_price
    logger.info(f"Filtering out data outside the price range {args.min_price} - {args.max_price}")
    df = df[(df['price'] >= args.min_price) & (df['price'] <= args.max_price)].copy()

    # Save the cleaned data to a CSV file
    clean_filename = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {clean_filename}")
    df.to_csv(clean_filename, index=False)

    # Create a new artifact and upload it to W&B
    logger.info(f"Uploading {clean_filename} to W&B as {args.output_artifact}")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(clean_filename)
    run.log_artifact(artifact)

    logger.info("Cleaned data artifact uploaded to W&B")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully qualified name of the input artifact in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to create in W&B",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., cleaned_data)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A brief description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to filter the data",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to filter the data",
        required=True
    )

    args = parser.parse_args()

    go(args)
