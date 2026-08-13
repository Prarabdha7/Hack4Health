import os
from pathlib import Path
from collections import Counter

from PIL import Image


DATASET_PATH = Path("data")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_images(root):
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_images(image_paths):
    sizes = Counter()
    formats = Counter()
    corrupted = []

    for path in image_paths:
        try:
            with Image.open(path) as image:
                sizes[image.size] += 1
                formats[image.format] += 1

        except Exception:
            corrupted.append(path)

    return sizes, formats, corrupted


def main():

    print("\n========== HACK4HEALTH DATASET INSPECTOR ==========\n")

    if not DATASET_PATH.exists():
        print(f"Dataset folder not found: {DATASET_PATH}")
        return

    image_paths = find_images(DATASET_PATH)

    print(f"Dataset location : {DATASET_PATH.resolve()}")
    print(f"Total images     : {len(image_paths)}")

    if not image_paths:
        print("\nNo supported image files were found.")
        print("Supported formats:", ", ".join(sorted(IMAGE_EXTENSIONS)))
        return

    sizes, formats, corrupted = inspect_images(image_paths)

    print("\n---------- Image Formats ----------")

    for image_format, count in formats.items():
        print(f"{image_format}: {count}")

    print("\n---------- Image Dimensions ----------")

    for size, count in sizes.most_common(10):
        print(f"{size[0]} × {size[1]} : {count} images")

    print("\n---------- Possible Classes ----------")

    class_counts = Counter(
        path.parent.name
        for path in image_paths
    )

    for class_name, count in class_counts.most_common():
        print(f"{class_name}: {count}")

    print("\n---------- Corrupted Images ----------")

    print(f"Corrupted/unreadable: {len(corrupted)}")

    if corrupted:
        for path in corrupted[:10]:
            print(f"  {path}")

        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")

    print("\n====================================================\n")


if __name__ == "__main__":
    main()