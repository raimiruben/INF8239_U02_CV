import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from inf8239_u02_cv.data import normalize_images
from inf8239_u02_cv.models import build_dense

ROOT = Path(__file__).resolve().parents[1]

(_, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x = normalize_images(x_test)

dense = build_dense(tf)
cnn = tf.keras.models.load_model(ROOT / "models" / "best_cnn.keras", compile=False)


def medir(model, nombre, repeticiones=7):
    model.predict(x, verbose=0)  # calentamiento, no se cuenta
    tiempos = []
    for _ in range(repeticiones):
        inicio = perf_counter()
        model.predict(x, verbose=0)
        tiempos.append(1000 * (perf_counter() - inicio) / len(x))
    print(f"{nombre}: mediana {np.median(tiempos):.4f} ms/imagen "
          f"(min {min(tiempos):.4f}, max {max(tiempos):.4f})")


medir(dense, "denso")
medir(cnn, "cnn")

with tempfile.TemporaryDirectory() as tmp:
    for nombre, model in [("denso", dense), ("cnn", cnn)]:
        ruta = os.path.join(tmp, nombre + ".keras")
        model.save(ruta)
        print(nombre, "archivo sin optimizador:", os.path.getsize(ruta), "bytes")