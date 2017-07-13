import tensorflow as tf

layers=tf.layers

def V_net(x,kernel_size,initializer,nonlinearity):
    ################################ Block1 ############################################################################
    tiled_Input = tf.tile(x, [1, 1, 1, 1, 8])
    layer_conv_in128_ch16 = layers.conv3d(inputs=x,filters=16,kernel_size=kernel_size,padding='same',kernel_initializer=initializer,
                                          name="weights_conv_in128_ch16")
    out_block_c11 = tf.nn.relu(tf.add(layer_conv_in128_ch16, tiled_Input))
    ############################### Stride1 #############################################################################
    layer_downConv_in128_out64 = layers.conv3d(inputs=out_block_c11,filters=32,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_downConv_in128_out64")
    ############################### Block2 #############################################################################
    layer_conv_in64_ch32 = layers.conv3d(inputs=layer_downConv_in128_out64,filters=32,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,
                                          name="weights_conv_in64_ch32")
    out_block_c12 = tf.nn.relu(tf.add(layer_conv_in64_ch32, layer_downConv_in128_out64))
    ################################ Stride2 ###########################################################################
    layer_downConv_in64_out32 = layers.conv3d(inputs=out_block_c12,filters=64,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_downConv_in64_out32")
    ############################## Block3 ##############################################################################
    layer_conv_in32_ch64 = layer_downConv_in64_out32
    for i in range(2):
        if i < 1:
            active=nonlinearity
        else:
            active=None
        layer_conv_in32_ch64 = layers.conv3d(inputs=layer_conv_in32_ch64,filters=64,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,activation=active,
                                          name="weights_conv_in64_ch32_"+str(i+1))
    out_block_c13 = tf.nn.relu(tf.add(layer_conv_in32_ch64, layer_downConv_in64_out32))
    ############################### Stride3 ############################################################################
    layer_downConv_in32_out16 = layers.conv3d(inputs=out_block_c13,filters=128,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_downConv_in32_out16")
    ############################### Block4 #############################################################################
    layer_conv_in16_ch128 = layer_downConv_in32_out16
    for i in range(3):
        if i < 2:
            active=nonlinearity
        else:
            active=None
        layer_conv_in16_ch128 = layers.conv3d(inputs=layer_conv_in16_ch128, filters=128, kernel_size=kernel_size,
                                             padding='same', kernel_initializer=initializer, activation=active,
                                             name="weights_conv_in16_ch128_" + str(i + 1))
    out_block_c14 = tf.nn.relu(tf.add(layer_conv_in16_ch128, layer_downConv_in32_out16))
    ################################ Stride4 ###########################################################################
    layer_downConv_in16_out8 = layers.conv3d(inputs=out_block_c14,filters=256,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_downConv_in16_out8")
    ################################# Block5 ###########################################################################
    layer_conv_in8_ch256 = layer_downConv_in16_out8
    for i in range(3):
        if i < 2:
            active=nonlinearity
        else:
            active=None
        layer_conv_in8_ch256 = layers.conv3d(inputs=layer_conv_in8_ch256, filters=256, kernel_size=kernel_size,
                                              padding='same', kernel_initializer=initializer, activation=active,
                                              name="weights_conv_in8_ch256_" + str(i + 1))
    out_block_c15 = tf.nn.relu(tf.add(layer_conv_in8_ch256, layer_downConv_in16_out8))
    ################################  deconv1 ##########################################################################
    layer_UpConv_in8_out16 = layers.conv3d_transpose(inputs=out_block_c15,filters=128,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_UpConv_in8_out16")
    layer_concat_in16 = tf.concat(axis=4, values=[layer_UpConv_in8_out16, out_block_c14])
    ################################ Block6 ############################################################################
    layer_conv_in16_ch256 = layer_concat_in16
    for i in range(3):
        if i < 2:
            active=nonlinearity
        else:
            active=None
            layer_conv_in16_ch256 = layers.conv3d(inputs=layer_conv_in16_ch256, filters=256, kernel_size=kernel_size,
                                             padding='same', kernel_initializer=initializer, activation=active,
                                             name="weights_conv_in16_ch256_" + str(i + 1))
    out_block_c24 = tf.nn.relu(tf.add(layer_conv_in16_ch256, layer_concat_in16))
    ############################## deconv2 #############################################################################
    layer_UpConv_in16_out32 = layers.conv3d_transpose(inputs=out_block_c24,filters=64,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_UpConv_in16_out32")
    layer_concat_in32 = tf.concat(axis=4, values=[layer_UpConv_in16_out32, out_block_c13])
    ############################## Block7 ##############################################################################
    layer_conv_in32_ch128 = layer_concat_in32
    for i in range(3):
        if i < 2:
            active=nonlinearity
        else:
            active=None
        layer_conv_in32_ch128 = layers.conv3d(inputs=layer_conv_in32_ch128,filters=128,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,activation=active,
                                          name="weights_conv_in32_ch128_"+str(i+1))
    out_block_c23 = tf.nn.relu(tf.add(layer_conv_in32_ch128, layer_concat_in32))
    ############################## deconv3 #############################################################################
    layer_UpConv_in32_out64 = layers.conv3d_transpose(inputs=out_block_c23,filters=32,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_UpConv_in32_out64")
    layer_concat_in64 = tf.concat(axis=4, values=[layer_UpConv_in32_out64, out_block_c12])
    ############################ Block8 ################################################################################
    layer_conv_in64_ch64 = layer_concat_in64
    for i in range(2):
        if i < 1:
            active=nonlinearity
        else:
            active=None
        layer_conv_in64_ch64 = layers.conv3d(inputs=layer_conv_in64_ch64,filters=64,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,activation=active,
                                          name="weights_conv_in64_ch64_"+str(i+1))
    out_block_c22 = tf.nn.relu(tf.add(layer_conv_in64_ch64, layer_concat_in64))
    ############################ deconv4 ###############################################################################
    layer_UpConv_in64_out128 = layers.conv3d_transpose(inputs=out_block_c22,filters=16,kernel_size=kernel_size,padding='same',
                                               strides=(2,2,2),kernel_initializer=initializer,
                                               activation=nonlinearity,name="weights_UpConv_in64_out128")
    layer_concat_in128 = tf.concat(axis=4, values=[layer_UpConv_in64_out128, out_block_c11])
    ########################### Block9 #################################################################################
    layer_conv_in128_ch32 = layers.conv3d(inputs=layer_concat_in128,filters=32,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,activation=None,
                                          name="weights_conv_in128_ch32")
    out_block_c21 = tf.nn.relu((tf.add(layer_conv_in128_ch32, layer_concat_in128)))
    ########################### Final Block ############################################################################
    layer_conv_in128_ch1 = layers.conv3d(inputs=out_block_c21,filters=1,kernel_size=kernel_size,
                                         padding='same',kernel_initializer=initializer,activation=None,
                                          name="weights_conv_in128_ch1")
    return layer_conv_in128_ch1
