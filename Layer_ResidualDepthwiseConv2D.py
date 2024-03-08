import tensorflow as tf

class ResidualDepthwiseConv2D(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, stride, padding, activation):
    super(ResidualDepthwiseConv2D, self).__init__()
    self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, activation=activation)
    self.pointwise = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), activation=activation)

  def call(self, inputs):
    x = self.depthwise(inputs)
    x = self.pointwise(x)

    x = self.depthwise(x)
    x = self.pointwise(x)

    x = tf.keras.layers.Add()([x, inputs])
    return x
