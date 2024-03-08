import tensorflow as tf

class DepthwiseSeparableConv2D(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, stride, padding, activation):
    super(DepthwiseSeparableConv2D, self).__init__()
    self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, activation=activation)
    self.pointwise = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), activation=activation)

  def call(self, input_tensor):
    x = self.depthwise(input_tensor)
    x = self.pointwise(x)
    return x
