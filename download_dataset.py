import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_NAME = "gpiosenka/100-bird-species"
OUTPUT_DIR = "dataset"

def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset...")
    api.dataset_download_files(
        DATASET_NAME,
        path=OUTPUT_DIR,
        unzip=True
    )

    print("Done! Dataset saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    download_dataset()