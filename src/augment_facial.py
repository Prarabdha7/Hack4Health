"""Generate a reproducible augmented TRAINING set only.
The validation/test split is created first and is never augmented, preventing leakage.
The original dataset is never modified.
"""
import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from config import IMAGE_DIR, EMOTION_CLASSES


def augment(im, rng):
    if rng.random() < 0.5: im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.35: im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.25: im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.90, 1.10))
    if rng.random() < 0.15: im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.6)))
    if rng.random() < 0.15:
        a=np.asarray(im,dtype=np.float32)
        a=np.clip(a+rng.normal(0,3.0,a.shape),0,255).astype(np.uint8)
        im=Image.fromarray(a)
    return im


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="data/augmented_images"); ap.add_argument("--copies",type=int,default=1); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    rng=np.random.default_rng(args.seed); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; total=0
    for emotion in EMOTION_CLASSES:
        folder=IMAGE_DIR/emotion
        paths=[p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".png",".jpg",".jpeg"}] if folder.is_dir() else []
        if not paths: continue
        train, temp=train_test_split(paths,test_size=.20,random_state=args.seed)
        val, test=train_test_split(temp,test_size=.50,random_state=args.seed)
        for split, items in (("train",train),("validation",val),("test",test)):
            for p in items: rows.append((str(p),emotion,split,"original"))
        dst=out/emotion; dst.mkdir(parents=True,exist_ok=True)
        for p in train:
            with Image.open(p).convert("L") as im:
                for i in range(args.copies):
                    q=dst/f"{p.stem}__aug{i+1}{p.suffix.lower()}"; augment(im.copy(),rng).save(q); rows.append((str(q),emotion,"train","augmented")); total+=1
    with open(out/"manifest.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["path","emotion","split","kind"]); w.writerows(rows)
    print(f"Generated {total} augmented training images in {out}")
    print(f"Manifest: {out/'manifest.csv'}")
    print("Validation and test originals are recorded but never augmented.")

if __name__=="__main__": main()
