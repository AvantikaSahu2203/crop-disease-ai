from pathlib import Path

DATASET_PATH = Path("datasets/raw/PlantVillage")

class_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

print("=" * 60)
print("PLANT DISEASE DATASET REPORT")
print("=" * 60)

total_images = 0

largest_class = ("", 0)
smallest_class = ("", float("inf"))

for folder in class_folders:

    image_files = []

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(folder.glob(ext))

    image_count = len(image_files)

    total_images += image_count

    if image_count > largest_class[1]:
        largest_class = (folder.name, image_count)

    if image_count < smallest_class[1]:
        smallest_class = (folder.name, image_count)

    print(f"{folder.name:<45}{image_count}")

print("\n" + "=" * 60)

print(f"Total Classes : {len(class_folders)}")
print(f"Total Images  : {total_images}")
print(f"Largest Class : {largest_class[0]} ({largest_class[1]})")
print(f"Smallest Class: {smallest_class[0]} ({smallest_class[1]})")

print("=" * 60)