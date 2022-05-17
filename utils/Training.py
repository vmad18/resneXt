from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Model
import tensorflow as tf
import numpy as np

true, false = True, False


def encode_data(y) -> np.ndarray:
    labelencoder = LabelEncoder()
    return np.asarray(to_categorical(labelencoder.fit_transform(y)), dtype="float32")


def training_loop(model: Model, loss, opt, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
                  encode: bool = true) -> Model:
    if encode:
        y = encode_data(y)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    acc_metric = keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):
        for step, d in enumerate(dataset):
            x0 = d[0]
            y0 = d[1]

            with tf.GradientTape() as tape:
                logits = model(x0, training=true)
                c_loss = loss(y0, logits)

            grads = tape.gradient(c_loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

            acc_metric.update_state(y0, logits)
            print(f"Accuracy: {acc_metric.result()}")
            print(f"Loss: {c_loss}")

    model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
    return model
