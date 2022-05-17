from keras import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, BatchNormalization, Activation, add

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


class IdentityBlock(Layer):

    def __init__(self, filters):
        super(IdentityBlock, self).__init__()

        f1, f2 = filters

        self.cl_1 = ConvLayer(f1)
        self.cl_2 = ConvLayer(f1, kernel=3, padding="same")
        self.cl_3 = ConvLayer(f2, activ=false)
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.cl_1(inputs)
        x = self.cl_2(x)
        x = self.cl_3(x)

        x = add([x, inputs])
        return self.activation(x)


class ConvBlock(Layer):

    def __init__(self, filters, strides=(1, 1)):
        super(ConvBlock, self).__init__()
        f1, f2 = filters

        self.cl_1 = ConvLayer(f1, strides=strides)
        self.cl_2 = ConvLayer(f1, kernel=3, padding="same")
        self.cl_3 = ConvLayer(f2, activ=false)

        self.shortcut = ConvLayer(f2, strides=strides, activ=false)
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.cl_1(inputs)
        x = self.cl_2(x)
        x = self.cl_3(x)

        cut = self.shortcut(inputs)
        x = add([x, cut])

        return self.activation(x)


class ResNet50(Model):

    def __init__(self, labels=10, out=true):
        super(ResNet50, self).__init__()

        self.labels = labels
        self.out = out

        self.zp3 = ZeroPadding2D(padding=3)
        self.conv = ConvLayer(64, kernel=7, strides=(2, 2))

        self.zp1 = ZeroPadding2D(padding=1)
        self.max_pool = MaxPooling2D(3, strides=(2, 2))

        self.cb1 = ConvBlock([64, 256])
        self.id1_1 = IdentityBlock([64, 256])
        self.id1_2 = IdentityBlock([64, 256])

        self.cb2 = ConvBlock([128, 512], strides=(2, 2))
        self.id2_1 = IdentityBlock([128, 512])
        self.id2_2 = IdentityBlock([128, 512])
        self.id2_3 = IdentityBlock([128, 512])

        self.cb3 = ConvBlock([256, 1024], strides=(2, 2))
        self.id3_1 = IdentityBlock([256, 1024])
        self.id3_2 = IdentityBlock([256, 1024])
        self.id3_3 = IdentityBlock([256, 1024])
        self.id3_4 = IdentityBlock([256, 1024])
        self.id3_5 = IdentityBlock([256, 1024])

        self.cb4 = ConvBlock([512, 2048])
        self.id4_1 = IdentityBlock([512, 2048])
        self.id4_2 = IdentityBlock([512, 2048])

        self.gap = GlobalAveragePooling2D()

        self.clf = Dense(self.labels, activation="softmax")

    def call(self, inputs):
        x = self.zp3(inputs)
        x = self.conv(x)

        x = self.zp1(x)
        x = self.max_pool(x)

        x = self.cb1(x)
        x = self.id1_1(x)
        x = self.id1_2(x)

        x = self.cb2(x)
        x = self.id2_1(x)
        x = self.id2_2(x)
        x = self.id2_3(x)

        x = self.cb3(x)
        x = self.id3_1(x)
        x = self.id3_2(x)
        x = self.id3_3(x)
        x = self.id3_4(x)
        x = self.id3_5(x)

        x = self.cb4(x)
        x = self.id4_1(x)
        x = self.id4_2(x)

        x = self.gap(x)

        if self.out: x = self.clf(x)
        return x
