import argparse
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from inf8239_u02_cv.data import normalize_images
from inf8239_u02_cv.models import build_cnn, build_dense

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    suffix = "" if args.epochs == 8 else f"_ep{args.epochs}"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    (x_all, y_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_all = normalize_images(x_all)
    x_test = normalize_images(x_test)
    x_train, y_train = x_all[:-6000], y_all[:-6000]
    x_valid, y_valid = x_all[-6000:], y_all[-6000:]

    histories, scores = {}, {}
    for name, factory in {"dense": build_dense, "cnn": build_cnn}.items():
        model = factory(tf)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=2,
                                                      restore_best_weights=True)]
        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                            epochs=args.epochs, batch_size=128,
                            callbacks=callbacks, verbose=2)
        histories[name] = {k: [float(v) for v in vals]
                           for k, vals in history.history.items()}
        predictions = model.predict(x_test, verbose=0).argmax(axis=1)
        scores[name] = float(f1_score(y_test, predictions, average="macro"))

    payload = {"epochs_requested": args.epochs, "seed": SEED,
               "history": histories, "f1_macro": scores}
    (reports / f"history{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for col, name in enumerate(["dense", "cnn"]):
        h = histories[name]
        epochs = range(1, len(h["loss"]) + 1)
        for row, metric in enumerate(["loss", "accuracy"]):
            ax = axes[row][col]
            ax.plot(epochs, h[metric], marker="o", label="entrenamiento")
            ax.plot(epochs, h["val_" + metric], marker="o", label="validacion")
            ax.set_title(f"{name} - {metric}")
            ax.set_xlabel("epoca")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("Curvas de entrenamiento y validacion (misma particion, semilla 42)")
    fig.tight_layout()
    fig.savefig(reports / f"training_curves{suffix}.png", dpi=150)

    official_path = reports / "cv_metrics.json"
    if args.epochs == 8 and official_path.exists():
        official = json.loads(official_path.read_text(encoding="utf-8"))
        for name in scores:
            diff = abs(scores[name] - official[name]["f1_macro"])
            print(name, "f1_macro:", scores[name],
                  "oficial:", official[name]["f1_macro"], "diferencia:", diff)
    for name, h in histories.items():
        print(name, "epocas ejecutadas:", len(h["loss"]),
              "val_loss final:", round(h["val_loss"][-1], 4),
              "val_accuracy final:", round(h["val_accuracy"][-1], 4))


if __name__ == "__main__":
    main()