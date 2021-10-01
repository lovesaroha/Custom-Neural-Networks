# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# VGG16 Network.
from tensorflow import keras


# Blocks.


class BlockTwoConv(keras.Model):
    def __init__(self, filters, kernel_size):
        super(BlockTwoConv, self).__init__(name="")
        # Convolutional layer.
        self.convOne = keras.layers.Conv2D(
            filters, kernel_size, padding="same", activation="relu")
        # Convolutional layer.
        self.convTwo = keras.layers.Conv2D(
            filters, kernel_size, padding="same", activation="relu")
        # Max pool.
        self.max_pool = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

    def call(self, input_tensor):
        x = self.convOne(input_tensor)
        x = self.convTwo(x)
        x = self.max_pool(x)
        return x

class BlockThreeConv(keras.Model):
    def __init__(self, filters, kernel_size):
        super(BlockThreeConv, self).__init__(name="")
        # Convolutional layer.
        self.convOne = keras.layers.Conv2D(
            filters, kernel_size, padding="same", activation="relu")
        # Convolutional layer.
        self.convTwo = keras.layers.Conv2D(
            filters, kernel_size, padding="same", activation="relu")
        # Convolutional layer.
        self.convThree = keras.layers.Conv2D(
            filters, kernel_size, padding="same", activation="relu")
        # Max pool.
        self.max_pool = keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")

    def call(self, input_tensor):
        x = self.convOne(input_tensor)
        x = self.convTwo(x)
        x = self.convThree(x)
        x = self.max_pool(x)
        return x

# VGG Network.

class VGG16(keras.Model):
    def __init__(self, total_classes):
        super(VGG16, self).__init__()
        # Blocks.
        self.blockOne = BlockTwoConv(64, 3)
        self.blockTwo = BlockTwoConv(128, 3)
        self.blockThree = BlockThreeConv(256, 3)
        self.blockFour = BlockThreeConv(512, 3) 
        self.blockFive = BlockThreeConv(512, 3)
        
        # Flatten.
        self.flatten = keras.layers.Flatten()
        # First dense.
        self.denseOne = keras.layers.Dense(4096 , activation="relu")
        # Second dense.
        self.denseTwo = keras.layers.Dense(4096 , activation="relu")
        # Output layer.
        self.output_layer = keras.layers.Dense(total_classes, activation="softmax")

    def call(self, inputs):
      x = self.blockOne(inputs)
      x = self.blockTwo(x)
      x = self.blockThree(x)
      x = self.blockFour(x)
      x = self.blockFive(x)
      x = self.flatten(x)
      x = self.denseOne(x)
      x = self.denseTwo(x)
      return self.output_layer(x)

