from tensorflow import keras
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from models import ResNeXt
from utils.Training import training_loop, encode_data
import numpy as np


def main() -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    resnext = ResNeXt(10)
    loss = CategoricalCrossentropy(from_logits=True)
    opt = Adam(learning_rate=1e-4)
    epochs = 20
    batch_size = 64

    # train ResNeXt model
    training_loop(
                    resnext,
                    loss,
                    opt,
                    np.asarray(x_train, dtype="float32"),
                    y_train,
                    epochs,
                    batch_size)

    print(resnext.evaluate(np.asarray(x_test), encode_data(y_test)))


if __name__ == '__main__':
    main()
