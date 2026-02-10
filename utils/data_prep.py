import json
import argparse
import logging
from pathlib import Path

# Setup logging for a more professional output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class YOLODatasetConverter:
    def __init__(self, source_path):
        self.source_path = Path(source_path)

    def create_dir(self, directory):
        """Creates directory if it doesn't exist."""
        Path(directory).mkdir(parents=True, exist_ok=True)

    def generate_manifests(self):
        """Generates .txt files containing paths to images for training/validation."""
        subfolders = [f for f in self.source_path.iterdir() if f.is_dir()]

        for folder in subfolders:
            dataset_name = folder.name
            # Grandparent directory logic
            output_root = folder.parent.parent
            manifest_file = output_root / f"{dataset_name}.txt"

            image_paths = sorted(folder.rglob("*.png"))

            with open(manifest_file, "w") as f:
                for img in image_paths:
                    # Creating a relative path for the manifest
                    relative_path = Path(".") / img.relative_to(output_root)
                    f.write(f"{relative_path}\n")

            logging.info(f"Manifest created: {manifest_file}")

    def convert_to_yolo(self):
        """Converts COCO JSON annotations to YOLO .txt format."""
        subfolders = [f for f in self.source_path.iterdir() if f.is_dir()]

        for folder in subfolders:
            annotation_file = folder / "_annotations.coco.json"

            if not annotation_file.exists():
                logging.warning(f"No annotations found in {folder}. Skipping.")
                continue

            # Define label directory (swapping 'images' for 'labels' in path)
            label_root = Path(str(folder).replace("images", "labels"))

            with open(annotation_file, "r") as j:
                data = json.load(j)

            # Map images by filename for quicker lookup
            images_map = {img["file_name"]: img for img in data["images"]}

            for img_path in folder.rglob("*.png"):
                # Extract location (equivalent to path_components[5] logic)
                location = img_path.parts[5] if len(img_path.parts) > 5 else "default"
                self.create_dir(label_root / location)

                # Matching logic with COCO JSON
                coco_key = f"leftImg8bit/{'/'.join(img_path.parts[4:])}"

                if coco_key in images_map:
                    img_info = images_map[coco_key]
                    h_orig, w_orig = img_info["height"], img_info["width"]

                    # Filter annotations for this specific image
                    img_annos = [
                        a
                        for a in data["annotations"]
                        if a["image_id"] == img_info["id"]
                    ]

                    label_file = label_root / location / f"{img_path.stem}.txt"

                    with open(label_file, "w") as out_f:
                        for anno in img_annos:
                            # COCO bbox: [x_min, y_min, width, height]
                            bx, by, bw, bh = anno["bbox"]

                            # Calculate YOLO format: normalized [center_x, center_y, width, height]
                            cx = (bx + (bw / 2)) / w_orig
                            cy = (by + (bh / 2)) / h_orig
                            nw = bw / w_orig
                            nh = bh / h_orig

                            # Shift category ID (COCO starts at 1, YOLO usually at 0)
                            class_id = anno["category_id"] - 1

                            out_f.write(
                                f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                            )

            logging.info(f"Successfully processed labels for: {folder.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Dataset Preprocessing Utility")
    parser.add_argument(
        "--source",
        type=str,
        default="./yolov7/customdata/fog/images/",
        help="Base path to the source images",
    )

    args = parser.parse_args()

    converter = YOLODatasetConverter(args.source)
    converter.generate_manifests()
    converter.convert_to_yolo()
