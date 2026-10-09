import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

from inf8239_u02_cv.data import normalize_images
from inf8239_u02_cv.models import build_dense

SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_all, y_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_all = normalize_images(x_all)
x_test = normalize_images(x_test)
x_train, y_train = x_all[:-6000], y_all[:-6000]
x_valid, y_valid = x_all[-6000:], y_all[-6000:]

model = build_dense(tf)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
          epochs=8, batch_size=128, callbacks=callbacks, verbose=2)

pred = model.predict(x_test, verbose=0).argmax(axis=1)
print("f1_macro:", f1_score(y_test, pred, average="macro"))
print(classification_report(y_test, pred, digits=3))