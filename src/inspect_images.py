import os
from PIL import Image

IMAGE_DIR = "data/images/Extracted_images"

counts = {}
dimensions = {}
corrupted = []

for emotion in sorted(os.listdir(IMAGE_DIR)):
    emotion_path = os.path.join(IMAGE_DIR, emotion)

    if not os.path.isdir(emotion_path):
        continue

    count = 0

    for filename in os.listdir(emotion_path):
        path = os.path.join(emotion_path, filename)

        if not os.path.isfile(path):
            continue

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                size = image.size

            dimensions[size] = dimensions.get(size, 0) + 1
            count += 1

        except Exception:
            corrupted.append(path)

    counts[emotion] = count

print("=" * 60)
print("FACIAL DATASET INSPECTION")
print("=" * 60)

print("\nClass counts:")

for emotion, count in counts.items():
    print(f"{emotion}: {count}")

print(f"\nTotal valid images: {sum(counts.values())}")

print("\nImage dimensions:")

for dimension, count in sorted(dimensions.items()):
    print(f"{dimension}: {count}")

print(f"\nCorrupted images: {len(corrupted)}")

if corrupted:
    print("\nFirst 10 corrupted files:")
    for path in corrupted[:10]:
        print(path)