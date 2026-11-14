
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# 1. Paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# 2. Classes
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

# 3. Settings
random.seed(42)
RATIOS = (0.70, 0.20, 0.10)

def main():

    images = {}
    for p in RAW.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images[p.name.lower()] = p

    xml_files = list(RAW.rglob("*.xml"))

    print("Images found:", len(images))
    print("XML files found:", len(xml_files))

    # 4. Match annotations and images
    records = []

    for xml in xml_files:
        try:
            root = ET.parse(xml).getroot()
            filename = root.findtext("filename")

            if not filename:
                filename = xml.stem + ".jpg"

            image_path = images.get(Path(filename).name.lower())

            if image_path is None:
                continue

            size = root.find("size")
            width = int(size.findtext("width"))
            height = int(size.findtext("height"))

            labels = []

            for obj in root.findall("object"):
                name = obj.findtext("name", "").strip().lower().replace("-", "_")

                if name not in CLASS_MAP:
                    continue

                box = obj.find("bndbox")
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))

                xmin = max(0, min(xmin, width))
                xmax = max(0, min(xmax, width))
                ymin = max(0, min(ymin, height))
                ymax = max(0, min(ymax, height))

                if xmax <= xmin or ymax <= ymin:
                    continue

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                class_id = CLASS_MAP[name]

                labels.append(
                    f"{class_id} {x_center:.6f} "
                    f"{y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )

            if labels:
                records.append({
                    "image": image_path,
                    "labels": labels,
                    "class": next(
                        (line.split()[0] for line in labels), None
                    )
                })

        except Exception as error:
            print("Skipped:", xml.name, error)

    # 5. Group by defect class
    groups = defaultdict(list)

    for record in records:
        groups[record["class"]].append(record)

    # 6. Split each class separately
    splits = {"train": [], "valid": [], "test": []}

    for class_id, items in groups.items():
        random.shuffle(items)

        n = len(items)
        train_end = int(n * RATIOS[0])
        valid_end = train_end + int(n * RATIOS[1])

        splits["train"].extend(items[:train_end])
        splits["valid"].extend(items[train_end:valid_end])
        splits["test"].extend(items[valid_end:])

    # 7. Save YOLO images and labels
    for split, items in splits.items():

        image_dir = ROOT / "data" / split / "images"
        label_dir = ROOT / "data" / split / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            image = item["image"]

            shutil.copy2(image, image_dir / image.name)

            label_file = label_dir / f"{image.stem}.txt"
            label_file.write_text(
                "\n".join(item["labels"]),
                encoding="utf-8"
            )

        print(f"{split.upper()}: {len(items)} images")

    # 8. Create data.yaml
    yaml = f"""path: {ROOT.as_posix()}
train: data/train/images
val: data/valid/images
test: data/test/images

nc: {len(CLASSES)}
names:
"""

    for class_id, name in enumerate(CLASSES):
        yaml += f"  {class_id}: {name}\n"

    (ROOT / "data.yaml").write_text(yaml, encoding="utf-8")

    print("\nDataset preparation completed!")
    print("Total valid image-label pairs:", len(records))
    print("YOLO configuration created:", ROOT / "data.yaml")


if __name__ == "__main__":
    main()
