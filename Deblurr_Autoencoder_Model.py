import tensorflow as tf
from Layer_DepthwiseSeparableConv2D import DepthwiseSeparableConv2D
from Layer_ResidualDepthwiseConv2D import ResidualDepthwiseConv2D
from Layer_CBAM import DMRFC

def Deblurr_Autoencoder(inputs):

  x = DepthwiseSeparableConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(inputs)
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  ### BOTTLE NECK
  x = DMRFC(channels=32, reduction_ratio=16, nb_layers=6)(x)
  ### BOTTLE NECK
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  x = ResidualDepthwiseConv2D(filters=32, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  x = DepthwiseSeparableConv2D(filters=3, kernel_size=(3,3), stride=1, padding="same", activation="leaky_relu")(x)
  output = tf.keras.layers.Add()([x, inputs])

  model = tf.keras.Model(inputs=inputs, outputs=output, name='autoencoder')
  return model