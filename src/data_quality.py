"""Dataset audit: counts, invalid images, exact duplicates and perceptual duplicates."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from config import IMAGE_DIR, AUDIO_DIR, REPORT_DIR, EMOTION_CLASSES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path, hash_size=8):
    img = Image.open(path).convert("L").resize((32, 32))
    a = np.asarray(img, dtype=np.float32)
    # DCT without scipy: use a small cosine basis.
    n = 32; k = 8
    x = np.arange(n)
    basis = np.cos(np.pi * (2 * x[:, None] + 1) * np.arange(k)[None, :] / (2*n))
    d = basis.T @ a @ basis
    low = d[:k, :k]
    med = np.median(low[1:])
    return "".join("1" if v > med else "0" for v in low.ravel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="store_true", help="also audit audio filenames/counts")
    args = parser.parse_args()
    report = {"image_classes": {}, "invalid_images": [], "exact_duplicate_groups": [], "perceptual_duplicate_groups": []}
    exact, perceptual = defaultdict(list), defaultdict(list)
    for emotion in EMOTION_CLASSES:
        folder = IMAGE_DIR / emotion
        paths = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}] if folder.is_dir() else []
        report["image_classes"][emotion] = len(paths)
        for p in paths:
            try:
                with Image.open(p) as im: im.verify()
                exact[sha256(p)].append(str(p)); perceptual[phash(p)].append(str(p))
            except Exception as e:
                report["invalid_images"].append({"path": str(p), "error": str(e)})
    report["exact_duplicate_groups"] = [v for v in exact.values() if len(v) > 1]
    report["perceptual_duplicate_groups"] = [v for v in perceptual.values() if len(v) > 1]
    if args.audio:
        audio = [p for p in AUDIO_DIR.rglob("*.wav")] if AUDIO_DIR.exists() else []
        report["audio_count"] = len(audio)
        report["audio_by_actor"] = {p.name: len(list(p.glob("*.wav"))) for p in sorted(AUDIO_DIR.iterdir()) if p.is_dir()} if AUDIO_DIR.exists() else {}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"exact_duplicate_groups", "perceptual_duplicate_groups"}}, indent=2))
    print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    print(f"Perceptual duplicate groups: {len(report['perceptual_duplicate_groups'])}")
    print(f"Saved: {out}")

if __name__ == "__main__": main()
