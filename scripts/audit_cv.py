import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u02_cv.data import normalize_images

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["camiseta", "pantal\u00f3n", "su\u00e9ter", "vestido", "abrigo",
         "sandalia", "camisa", "zapatilla", "bolso", "bot\u00edn"]


def load_partitions():
    (x_all, y_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return {
        "train": (x_all[:-6000], y_all[:-6000]),
        "valid": (x_all[-6000:], y_all[-6000:]),
        "test": (x_test, y_test),
    }


def fingerprint(x, y):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()[:16]


def image_hashes(x):
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in x]


parts = load_partitions()
report = {"classes": NAMES, "partitions": {}}
hash_sets = {}
for name, (x, y) in parts.items():
    counts = np.bincount(y, minlength=10)
    hashes = image_hashes(x)
    hash_sets[name] = set(hashes)
    report["partitions"][name] = {
        "images": int(len(x)),
        "image_shape": [int(v) for v in x.shape[1:]],
        "dtype": str(x.dtype),
        "pixel_min": int(x.min()),
        "pixel_max": int(x.max()),
        "class_counts": {NAMES[i]: int(c) for i, c in enumerate(counts)},
        "class_proportions": {NAMES[i]: round(float(c) / len(y), 4)
                              for i, c in enumerate(counts)},
        "duplicate_images_within": int(len(hashes) - len(hash_sets[name])),
        "fingerprint": fingerprint(x, y),
    }

report["overlap_identical_images"] = {
    "valid_and_train": len(hash_sets["valid"] & hash_sets["train"]),
    "test_and_train": len(hash_sets["test"] & hash_sets["train"]),
    "test_and_valid": len(hash_sets["test"] & hash_sets["valid"]),
}

normalized = normalize_images(parts["train"][0])
report["normalized_train"] = {
    "shape": [int(v) for v in normalized.shape],
    "dtype": str(normalized.dtype),
    "min": float(normalized.min()),
    "max": float(normalized.max()),
}

again = load_partitions()
report["partition_reproducible"] = all(
    fingerprint(*again[name]) == report["partitions"][name]["fingerprint"]
    for name in parts
)

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "audit_cv.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)

fig, ax = plt.subplots(figsize=(9, 4))
width = 0.27
positions = np.arange(10)
for k, name in enumerate(["train", "valid", "test"]):
    values = [report["partitions"][name]["class_proportions"][n] for n in NAMES]
    ax.bar(positions + (k - 1) * width, values, width, label=name)
ax.set_xticks(positions)
ax.set_xticklabels(NAMES, rotation=30, ha="right")
ax.set_ylabel("proporcion")
ax.set_title("Distribucion de clases por particion")
ax.set_ylim(0, 0.13)
ax.legend(ncol=3, loc="upper center")
fig.tight_layout()
fig.savefig(ROOT / "reports" / "audit_class_distribution.png", dpi=150)

for name, info in report["partitions"].items():
    print(name, info["images"], info["image_shape"], info["dtype"],
          info["pixel_min"], info["pixel_max"],
          "duplicados:", info["duplicate_images_within"],
          "huella:", info["fingerprint"])
    print("  clases:", info["class_counts"])
print("solapamiento (imagenes identicas):", report["overlap_identical_images"])
print("normalizadas:", report["normalized_train"])
print("particion reproducible:", report["partition_reproducible"])