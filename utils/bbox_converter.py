import argparse
import json
import os


def load_json(path):
    """Utility to load JSON data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def convert_bbox(bbox, input_fmt, output_fmt):
    """
    Logic for coordinate conversion.
    TODO: Implement YOLO and specific COCO/VOC edge cases tomorrow.
    """
    # Current implementation: COCO [x, y, w, h] -> VOC [xmin, ymin, xmax, ymax]
    if input_fmt == "COCO" and output_fmt == "VOC":
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    # Placeholder for other formats
    return bbox


def format_converter(args):
    print(f"--- Starting conversion: {args.input_format} to {args.output_format} ---")

    data = load_json(args.input)
    reference = load_json(args.refer_path) if args.convert_image_id else None

    new_data = {}

    for obj in data:
        # 1. Handle Image ID / Filename Mapping
        image_key = str(obj["image_id"])
        if args.convert_image_id and reference:
            # TODO: Optimize this lookup with a dictionary tomorrow for speed
            for img in reference.get("images", []):
                if img["id"] == obj["image_id"]:
                    image_key = img["file_name"]
                    break
        else:
            image_key = f"{image_key}.png"

        # 2. Convert Bounding Box
        converted_bbox = convert_bbox(
            obj["bbox"], args.input_format, args.output_format
        )

        # 3. Organize Data
        if image_key not in new_data:
            new_data[image_key] = {"boxes": [], "labels": [], "scores": []}

        new_data[image_key]["labels"].append(obj["category_id"] + 1)
        new_data[image_key]["boxes"].append(converted_bbox)
        new_data[image_key]["scores"].append(obj.get("score", 0.0))

    # 4. Save Output
    with open(args.output, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"--- Conversion complete! Saved to: {args.output} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Format Converter")

    # Path Arguments
    parser.add_argument(
        "--input", type=str, default="results.json", help="Input JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="converted_results.json", help="Output JSON file"
    )

    # Format Arguments
    parser.add_argument(
        "--input-format", type=str, default="COCO", choices=["COCO", "YOLO", "VOC"]
    )
    parser.add_argument(
        "--output-format", type=str, default="VOC", choices=["COCO", "YOLO", "VOC"]
    )

    # Mapping Arguments
    parser.add_argument(
        "--convert-image-id",
        action="store_true",
        help="Convert numeric IDs to filenames",
    )
    parser.add_argument(
        "--refer-path",
        type=str,
        default="annotations.json",
        help="Reference COCO file for ID mapping",
    )

    args = parser.parse_args()
    format_converter(args)
