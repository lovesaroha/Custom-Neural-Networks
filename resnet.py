# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Residual Network.
from tensorflow import keras


# Block.


class Block(keras.Model):
    def __init__(self, filters, kernel_size):
        super(Block, self).__init__(name="")
        # Convolutional layer.
        self.convOne = keras.layers.Conv2D(
            filters, kernel_size, padding="same")
        # Batch normalize.
        self.batchNormalizeOne = keras.layers.BatchNormalization()

        # Convolutional layer.
        self.convTwo = keras.layers.Conv2D(
            filters, kernel_size, padding="same")
        # Batch normalize.
        self.batchNormalizeTwo = keras.layers.BatchNormalization()

        # Activation.
        self.activation = keras.layers.Activation("relu")
        self.add = keras.layers.Add()

    def call(self, input_tensor):
        x = self.convOne(input_tensor)
        x = self.batchNormalizeOne(x)
        x = self.activation(x)

        x = self.convTwo(x)
        x = self.batchNormalizeTwo(x)
        x = self.activation(x)

        x = self.add([x, input_tensor])
        x = self.activation(x)
        return x


# Residual Network.

class ResNet(keras.Model):
    def __init__(self, total_classes):
        super(ResNet, self).__init__()
        # Convolutional layer.
        self.conv = keras.layers.Conv2D(
            64, (7,7), padding="same")
        # Batch normalize.
        self.batchNormalize = keras.layers.BatchNormalization()
        # Activation.
        self.activation = keras.layers.Activation("relu")
        # Max pool.
        self.max_pool = keras.layers.MaxPool2D((3, 3))

        # Blocks.
        self.blockOne = Block(64, 3)
        self.blockTwo = Block(64, 3)

        # Global pool.
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        # Output layer.
        self.output_layer = keras.layers.Dense(total_classes, activation="softmax")

    def call(self, inputs):
      x = self.conv(inputs)
      x = self.batchNormalize(x)
      x = self.activation(x)
      x = self.max_pool(x)
      x = self.blockOne(x)
      x = self.blockTwo(x)
      x = self.global_pool(x)
      return self.output_layer(x)


