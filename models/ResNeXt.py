from keras import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, BatchNormalization, Activation, add, concatenate

true, false = True, False


class ConvLayer(Layer):

    def __init__(self, filters, kernel=1, strides=(1, 1), padding="valid", activ=true):
        super(ConvLayer, self).__init__()

        self.activ = activ
        self.conv = Conv2D(
                            filters,
                            kernel,
                            strides=strides,
                            padding=padding,
                            kernel_initializer="he_normal")
        self.bn = BatchNormalization(axis=3)
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activ: x = self.activation(x)
        return x


class StackConv(Layer):

    def __init__(self, filters: int, kernel: int, padding: str, strides: int, in_channels: int, cardinality: int):
        super(StackConv, self).__init__()

        self.cardinality = cardinality

        self.f_s: int = filters / cardinality  # filters per cardinality
        self.p_f: int = in_channels / cardinality  # previous channels per cardinality

        self.convs = []  # stacked convolutions

        for i in range(cardinality):
            self.convs.append(
                Conv2D(
                        self.f_s,
                        kernel,
                        strides=strides,
                        padding=padding,
                        kernel_initializer="he_normal")
                )

    def call(self, inputs):
        maps = []

        for i in range(self.cardinality):
            maps.append(
                self.convs[i](inputs[:, :, :, int(self.p_f * i):int((i + 1) * self.p_f)])
            )
        return concatenate(maps, -1)


class ResNeXtBlock(Layer):

    def __init__(self, filters, cardinality, strides=(2, 2)):
        super(ResNeXtBlock, self).__init__()
        f1, f2 = filters
        self.c1 = ConvLayer(f1)
        self.sc = StackConv(f1, kernel=3, padding="same", strides=strides, in_channels=f1, cardinality=cardinality)
        self.c2 = ConvLayer(f2, activ=false)

        self.shortcut = ConvLayer(f2, strides=strides, activ=false)
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.sc(x)
        x = self.c2(x)

        cut = self.shortcut(inputs)
        x = add([x, cut])
        return self.activation(x)


class RepeatBlock(Layer):

    def __init__(self, filters, cardinality, count: int):
        super(RepeatBlock, self).__init__()

        self.blocks = []
        for i in range(count):
            self.blocks.append(ResNeXtBlock(filters, cardinality))

    def call(self, inputs):
        x = self.blocks[0](inputs)
        for i in range(1, len(self.blocks)):
            x = self.blocks[i](x)
        return x


class ResNeXt50(Model):

    def __init__(self, labels=10, out=true):
        super(ResNeXt50, self).__init__()
        self.labels = labels
        self.out = out

        self.zp3 = ZeroPadding2D(padding=3)
        self.conv = ConvLayer(64, kernel=7, strides=(2, 2))

        self.zp1 = ZeroPadding2D(padding=1)
        self.max_pool = MaxPooling2D(3, strides=(2, 2))

        self.b1 = RepeatBlock([128, 256], 32, 3)
        self.b2 = RepeatBlock([256, 512], 32, 3)
        self.b3 = RepeatBlock([512, 1024], 32, 3)
        self.b4 = RepeatBlock([1024, 2048], 32, 3)

        self.gap = GlobalAveragePooling2D()
        self.clf = Dense(self.labels, activation="softmax")

    def call(self, inputs):
        x = self.zp3(inputs)
        x = self.conv(x)

        x = self.zp1(x)
        x = self.max_pool(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        x = self.gap(x)
        if self.out:
            x = self.clf(x)
        return x
