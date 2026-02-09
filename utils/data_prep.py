import os
import glob
import json
import argparse


def create_folder(path):
    """Safely creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preprocessing Utility")
    parser.add_argument(
        "--source-folder-path",
        type=str,
        default="./data/images/",
        help="Path to the images folder",
    )
    args = parser.parse_args()
    print("Environment initialized.")
