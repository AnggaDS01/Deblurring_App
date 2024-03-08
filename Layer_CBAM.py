import tensorflow as tf

class CBAM(tf.keras.layers.Layer):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, gate_channels=32, reduction_ratio=8):
        super().__init__()
        self.gate_channels = gate_channels
        self.ratio = reduction_ratio
        self.shared_layer_one = tf.keras.layers.Dense(gate_channels // reduction_ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(gate_channels,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.multiply = tf.keras.layers.Multiply()

    def build(self, input_shape):
        # input_shape is a tuple of (batch_size, height, width, channels)
        self.channel_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1
        self.channel = input_shape[self.channel_axis]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        if tf.keras.backend.image_data_format() == "channels_first":
            self.permute = tf.keras.layers.Permute((3, 1, 2))
        else:
            self.permute = None
        self.reshape = tf.keras.layers.Reshape((1, 1, self.channel))

    def call(self, inputs):
        cbam_feature = self.channel_attention(inputs)
        return cbam_feature

    def channel_attention(self, input_feature):
        avg_pool = self.avg_pool(input_feature)
        avg_pool = self.reshape(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        max_pool = self.max_pool(input_feature)
        max_pool = self.reshape(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        cbam_feature = self.add([avg_pool, max_pool])
        cbam_feature = self.activation(cbam_feature)
        if self.permute is not None:
            cbam_feature = self.permute(cbam_feature)
        return self.multiply([input_feature, cbam_feature])

class R(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(R, self).__init__()
        self.channels = channels

    def build(self, input_shape):
        self.conv_reduced = tf.keras.layers.Conv2D(self.channels, 1)
        self.conv1 = tf.keras.layers.Conv2D(self.channels, (3, 3), padding='same', dilation_rate=1)
        self.conv2 = tf.keras.layers.Conv2D(self.channels, (3, 3), padding='same', dilation_rate=3)
        self.conv3 = tf.keras.layers.Conv2D(self.channels, (3, 3), padding='same', dilation_rate=5)
        self.conv4 = tf.keras.layers.Conv2D(self.channels, (3, 3), padding='same', dilation_rate=7)
        super(R, self).build(input_shape)

    def call(self, input_tensor):
        reduced = self.conv_reduced(input_tensor)
        conv1 = self.conv1(reduced)
        conv2 = self.conv2(reduced)
        conv3 = self.conv3(reduced)
        conv4 = self.conv4(reduced)
        cat = tf.keras.layers.Concatenate()([conv1, conv2, conv3, conv4, input_tensor])
        return cat

class MultireceptiveChannelBlocks(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio):
      super(MultireceptiveChannelBlocks, self).__init__()
      self.channels = channels
      self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
      self.r_block = R(self.channels)
      self.cbam_block = CBAM(gate_channels=self.channels*5, reduction_ratio=self.reduction_ratio)
      self.conv = tf.keras.layers.Conv2D(self.channels, kernel_size=(1, 1))
      super(MultireceptiveChannelBlocks, self).build(input_shape)  # Memanggil metode build dari kelas induk

    def call(self, input_tensor):
      r_output = self.r_block(input_tensor)
      cbam_output = self.cbam_block(r_output)
      output = self.conv(cbam_output)
      return output

class DMRFC(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, nb_layers=4):
      super(DMRFC, self).__init__()
      self.channels = channels
      self.reduction_ratio = reduction_ratio
      self.nb_layers = nb_layers

    def build(self, input_shape):
      self.mrc_blocks = [MultireceptiveChannelBlocks(self.channels, self.reduction_ratio) for _ in range(self.nb_layers)]
      self.concat = tf.keras.layers.Concatenate()
      self.conv = tf.keras.layers.Conv2D(self.channels, kernel_size=(1, 1))
      super(DMRFC, self).build(input_shape)

    def call(self, input_tensor):
      concat_feat = input_tensor
      for mrc_block in self.mrc_blocks:
        x = mrc_block(input_tensor)
        concat_feat = self.concat([concat_feat, x])
      output = self.conv(concat_feat)
      return output