import argparse
import json
import sys
import os


def load_data(file_path):
    """Helper to load JSON with basic error handling."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    with open(file_path, "r") as f:
        return json.load(f)


def format_converter(args):
    # Load primary detection results
    raw_data = load_data(args.input)

    # Setup for Image ID conversion
    id_to_filename = {}
    if args.convert_image_id:
        reference = load_data(args.refer_path)
        # Create a lookup map for O(1) speed instead of nested loops
        id_to_filename = {
            img["id"]: img["file_name"] for img in reference.get("images", [])
        }

    converted_results = {}

    for entry in raw_data:
        image_id = entry["image_id"]
        category_id = entry["category_id"] + 1
        bbox = list(entry["bbox"])  # Ensure it's a list for mutability
        score = entry.get("score", 0)

        # Handle Image ID naming
        if args.convert_image_id:
            filename = id_to_filename.get(image_id, f"unknown_{image_id}.png")
        else:
            filename = f"{image_id}.png"

        # Coordinate Transformation: COCO [x, y, w, h] -> VOC [xmin, ymin, xmax, ymax]
        if args.input_format == "COCO" and args.output_format == "VOC":
            # bbox[2] (width) becomes x_max, bbox[3] (height) becomes y_max
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]

        # Structure the dictionary output
        if filename not in converted_results:
            converted_results[filename] = {"boxes": [], "labels": [], "scores": []}

        converted_results[filename]["labels"].append(category_id)
        converted_results[filename]["boxes"].append(bbox)
        converted_results[filename]["scores"].append(score)

    # Save output with indentation for readability
    with open(args.output, "w") as f:
        json.dump(converted_results, f, indent=4)

    print(f"Successfully converted {len(raw_data)} detections to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="DETR/YOLO Result Format Converter")

    # File Paths
    parser.add_argument(
        "--input",
        type=str,
        default="yolov7_results.json",
        help="Input JSON file from detector",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="yolov7_results_converted.json",
        help="Filename for converted output",
    )
    parser.add_argument(
        "--refer-path",
        type=str,
        default="valid/annotations.coco.json",
        help="Reference COCO file for filename mapping",
    )

    # Configuration
    parser.add_argument(
        "--input-format", type=str, default="COCO", help="Input bbox format"
    )
    parser.add_argument(
        "--output-format", type=str, default="VOC", help="Output bbox format"
    )
    parser.add_argument(
        "--convert-image-id",
        action="store_true",
        help="Convert numeric IDs to filenames using reference path",
    )

    args = parser.parse_args()
    format_converter(args)


if __name__ == "__main__":
    main()
