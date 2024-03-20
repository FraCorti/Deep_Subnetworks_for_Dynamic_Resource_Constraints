import os

import numpy as np
import tensorflow as tf

from utils.batch_normalization import Reds_BatchNormalizationBase
from utils.convolution import Reds_2DConvolution_Standard
from utils.ds_convolution import Reds_DepthwiseConv2D
from utils.global_average import Reds_GlobalAveragePooling2D
from utils.linear import Linear_Adaptive
from utils.relu import Reds_ReLU
from utils.reshape import Reds_Reshape
from utils.visual_wake_words import MobileNet
from utils.zero_padding import Reds_ZeroPadding2D


def load_mobilenet_v1_visual_wake_words_leaky_relu(args, alpha=1.0, visual_wake_words=True, input_image_size=96):
    input_shape = (input_image_size, input_image_size, args.input_shape_channels)

    pretrained_model = tf.keras.models.load_model(
        f"{os.getcwd()}/visual_wake_words/visualwakeup_aesd/pretrained_model_input_preprocessed/vww_mobilenet_1.0_{input_image_size}_{input_image_size}_coco2014_0.9_leaky.keras")

    model = MobileNet(
        input_shape=input_shape, alpha=alpha, include_top=False, visual_wake_words=visual_wake_words, weights=None
    )

    for trained_variable_index in range(len(pretrained_model.variables)):
        model.variables[trained_variable_index].assign(pretrained_model.variables[trained_variable_index])

    return model


def load_mobilenet_v1_visual_wake_words(args, alpha=1.0, visual_wake_words=True):
    input_shape = (args.input_shape_height, args.input_shape_width, args.input_shape_channels)

    pretrained_model = tf.keras.models.load_model(
        f"{os.getcwd()}/visual_wake_words/visualwakeup_aesd/pretrained_model_input_preprocessed/vww_mobilenet_1.0_96_96_coco2014_0.9_new.keras")

    model = MobileNet(
        input_shape=input_shape, alpha=alpha, include_top=False, visual_wake_words=visual_wake_words, weights=None
    )

    for trained_variable_index in range(len(pretrained_model.variables)):
        model.variables[trained_variable_index].assign(pretrained_model.variables[trained_variable_index])

    return model


class Reds_MobilenetV1_Leaky(tf.keras.Model):
    def __init__(self, classes, subnetworks_number=4,
                 batch_dimensions=4,
                 use_bias=False,
                 debug=False):
        super(Reds_MobilenetV1_Leaky, self).__init__()

        self.subnetworks_number = subnetworks_number
        self.classes = classes

        # first standard convolution
        self.standard_convolution = Reds_2DConvolution_Standard(in_channels=3, out_channels=32,
                                                                kernel_size=(3, 3),
                                                                batch_dimensions=batch_dimensions,
                                                                use_bias=use_bias,
                                                                strides=(2,
                                                                         2),
                                                                debug=debug, padding='same')
        self.batch_norm1 = Reds_BatchNormalizationBase(fused=False)
        self.relu1 = tf.keras.layers.LeakyReLU()

        # first block
        self.depthwise1 = Reds_DepthwiseConv2D(in_channels=32, kernel_size=(3, 3),
                                               use_bias=use_bias,
                                               strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm2 = Reds_BatchNormalizationBase(fused=False)
        self.relu2 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv1 = Reds_2DConvolution_Standard(in_channels=32, out_channels=64,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm3 = Reds_BatchNormalizationBase(fused=False)
        self.relu3 = tf.keras.layers.LeakyReLU()

        # second block
        self.depthwise2 = Reds_DepthwiseConv2D(in_channels=64, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm4 = Reds_BatchNormalizationBase(fused=False)
        self.relu4 = tf.keras.layers.LeakyReLU()

        self.pointwise_conv2 = Reds_2DConvolution_Standard(in_channels=64, out_channels=128,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm5 = Reds_BatchNormalizationBase(fused=False)
        self.relu5 = tf.keras.layers.LeakyReLU()

        # third block
        self.depthwise3 = Reds_DepthwiseConv2D(in_channels=128, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm6 = Reds_BatchNormalizationBase(fused=False)
        self.relu6 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv3 = Reds_2DConvolution_Standard(in_channels=128, out_channels=128,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm7 = Reds_BatchNormalizationBase(fused=False)
        self.relu7 = tf.keras.layers.LeakyReLU()

        # fourth block
        self.depthwise4 = Reds_DepthwiseConv2D(in_channels=128, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm8 = Reds_BatchNormalizationBase(fused=False)
        self.relu8 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv4 = Reds_2DConvolution_Standard(in_channels=128, out_channels=256,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm9 = Reds_BatchNormalizationBase(fused=False)
        self.relu9 = tf.keras.layers.LeakyReLU()

        # fifth block
        self.depthwise5 = Reds_DepthwiseConv2D(in_channels=256, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm10 = Reds_BatchNormalizationBase(fused=False)
        self.relu10 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv5 = Reds_2DConvolution_Standard(in_channels=256, out_channels=256,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm11 = Reds_BatchNormalizationBase(fused=False)
        self.relu11 = tf.keras.layers.LeakyReLU()

        # sixth block
        self.depthwise6 = Reds_DepthwiseConv2D(in_channels=256, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm12 = Reds_BatchNormalizationBase(fused=False)
        self.relu12 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv6 = Reds_2DConvolution_Standard(in_channels=256, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm13 = Reds_BatchNormalizationBase(fused=False)
        self.relu13 = tf.keras.layers.LeakyReLU()

        # seventh block
        self.depthwise7 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm14 = Reds_BatchNormalizationBase(fused=False)
        self.relu14 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv7 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm15 = Reds_BatchNormalizationBase(fused=False)
        self.relu15 = tf.keras.layers.LeakyReLU()

        # eighth block
        self.depthwise8 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm16 = Reds_BatchNormalizationBase(fused=False)
        self.relu16 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv8 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm17 = Reds_BatchNormalizationBase(fused=False)
        self.relu17 = tf.keras.layers.LeakyReLU()

        # ninth block
        self.depthwise9 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm18 = Reds_BatchNormalizationBase(fused=False)
        self.relu18 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv9 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm19 = Reds_BatchNormalizationBase(fused=False)
        self.relu19 = tf.keras.layers.LeakyReLU()

        # tenth block
        self.depthwise10 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm20 = Reds_BatchNormalizationBase(fused=False)
        self.relu20 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv10 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm21 = Reds_BatchNormalizationBase(fused=False)
        self.relu21 = tf.keras.layers.LeakyReLU()

        # eleventh block
        self.depthwise11 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm22 = Reds_BatchNormalizationBase(fused=False)
        self.relu22 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv11 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm23 = Reds_BatchNormalizationBase(fused=False)
        self.relu23 = tf.keras.layers.LeakyReLU()

        # twelfth block
        self.depthwise12 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(2, 2),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm24 = Reds_BatchNormalizationBase(fused=False)
        self.relu24 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv12 = Reds_2DConvolution_Standard(in_channels=512, out_channels=1024,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm25 = Reds_BatchNormalizationBase(fused=False)
        self.relu25 = tf.keras.layers.LeakyReLU()

        # thirteenth block
        self.depthwise13 = Reds_DepthwiseConv2D(in_channels=1024, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm26 = Reds_BatchNormalizationBase(fused=False)
        self.relu26 = tf.keras.layers.LeakyReLU()
        self.pointwise_conv13 = Reds_2DConvolution_Standard(in_channels=1024, out_channels=1024,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm27 = Reds_BatchNormalizationBase(fused=False)
        self.relu27 = tf.keras.layers.LeakyReLU()

        self.head_classification = Linear_Adaptive(in_features=1024, out_features=classes,
                                                   debug=debug, use_bias=True)
        self.dense_model_trainable_weights = 0

    def finetune_batch_normalization(self):
        """
        Used to finetune the Batch Normalization layers while freezing all
        the other layers
        """
        for layer in self.layers:
            if isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = True
            else:
                layer.trainable = False

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def get_subnetwork_parameters_percentage(self, subnetwork_index):
        """
        If the input feature are not registered perform a forward pass on the adaptive layers to compute them, return the
        number of trainable parameters used by the subnetwork
        @param subnetwork_index:
        @return:
        """

        if self.dense_model_trainable_weights == 0:
            self.dense_model_trainable_weights = sum(
                [np.prod(tensor.shape) if not isinstance(tensor, Reds_BatchNormalizationBase) else 0 for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers[:-1]:

            if isinstance(layer, Reds_BatchNormalizationBase) or isinstance(layer,
                                                                            tf.keras.layers.Reshape) or isinstance(
                layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.LeakyReLU) or isinstance(layer,
                                                                                                           tf.keras.layers.Flatten) or isinstance(
                layer,
                tf.keras.layers.AveragePooling2D):
                continue
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def get_model_name(self):
        return "mobilenet_v1_vww"

    def compute_inference_estimations(self, input_shape=(1, 96, 96, 3)):

        layers_filters_macs = []
        layers_filters_byte = []
        layers_filters_activation_map_byte = []

        inputs = tf.ones(input_shape, dtype=tf.dtypes.float32)

        for layer in self.layers[:len(self.layers) - 1]:

            if isinstance(layer, Reds_DepthwiseConv2D) or isinstance(layer, Reds_2DConvolution_Standard):
                inputs, macs, filters_byte_memory, filters_activation_maps_memory = layer.compute_layer_inference_estimations(
                    inputs=inputs)
                layers_filters_macs.append(macs)
                layers_filters_byte.append(filters_byte_memory)
                layers_filters_activation_map_byte.append(filters_activation_maps_memory)

        return layers_filters_macs, layers_filters_byte, layers_filters_activation_map_byte

    def build(self, input_shape):
        """
        Initialize model's layers weights and biases
        """
        self.set_subnetworks_number(subnetworks_number=1)
        init_input = tf.ones(
            input_shape,
            dtype=tf.dtypes.float32,
        )
        init_input = [init_input for _ in range(1)]

        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase):
                _ = layer.build(init_input[0].shape)
            elif isinstance(layer, Reds_2DConvolution_Standard):
                init_input = layer(init_input)
            elif isinstance(layer, Reds_DepthwiseConv2D):
                init_input = layer(init_input)
            elif isinstance(layer, Linear_Adaptive):
                init_input = [tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(input) for input in init_input]
                init_input = [tf.keras.layers.Flatten()(input) for input in init_input]
                init_input = layer(init_input)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs = [inputs for _ in range(self.subnetworks_number)]

        # first standard convolution
        inputs = self.standard_convolution(inputs)
        inputs = self.batch_norm1(inputs)
        inputs = [self.relu1(input) for input in inputs]

        # first block
        inputs = self.depthwise1(inputs)
        inputs = self.batch_norm2(inputs)
        inputs = [self.relu2(input) for input in inputs]
        inputs = self.pointwise_conv1(inputs)
        inputs = self.batch_norm3(inputs)
        inputs = [self.relu3(input) for input in inputs]

        # second block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise2(inputs)
        inputs = self.batch_norm4(inputs)
        inputs = [self.relu4(input) for input in inputs]
        inputs = self.pointwise_conv2(inputs)
        inputs = self.batch_norm5(inputs)
        inputs = [self.relu5(input) for input in inputs]

        # third block
        inputs = self.depthwise3(inputs)
        inputs = self.batch_norm6(inputs)
        inputs = [self.relu6(input) for input in inputs]
        inputs = self.pointwise_conv3(inputs)
        inputs = self.batch_norm7(inputs)
        inputs = [self.relu7(input) for input in inputs]

        # fourth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise4(inputs)
        inputs = self.batch_norm8(inputs)
        inputs = [self.relu8(input) for input in inputs]
        inputs = self.pointwise_conv4(inputs)
        inputs = self.batch_norm9(inputs)
        inputs = [self.relu9(input) for input in inputs]

        # fifth block
        inputs = self.depthwise5(inputs)
        inputs = self.batch_norm10(inputs)
        inputs = [self.relu10(input) for input in inputs]
        inputs = self.pointwise_conv5(inputs)
        inputs = self.batch_norm11(inputs)
        inputs = [self.relu11(input) for input in inputs]

        # sixth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise6(inputs)
        inputs = self.batch_norm12(inputs)
        inputs = [self.relu12(input) for input in inputs]
        inputs = self.pointwise_conv6(inputs)
        inputs = self.batch_norm13(inputs)
        inputs = [self.relu13(input) for input in inputs]

        # seventh block
        inputs = self.depthwise7(inputs)
        inputs = self.batch_norm14(inputs)
        inputs = [self.relu14(input) for input in inputs]
        inputs = self.pointwise_conv7(inputs)
        inputs = self.batch_norm15(inputs)
        inputs = [self.relu15(input) for input in inputs]

        # eighth block
        inputs = self.depthwise8(inputs)
        inputs = self.batch_norm16(inputs)
        inputs = [self.relu16(input) for input in inputs]
        inputs = self.pointwise_conv8(inputs)
        inputs = self.batch_norm17(inputs)
        inputs = [self.relu17(input) for input in inputs]

        # ninth block
        inputs = self.depthwise9(inputs)
        inputs = self.batch_norm18(inputs)
        inputs = [self.relu18(input) for input in inputs]
        inputs = self.pointwise_conv9(inputs)
        inputs = self.batch_norm19(inputs)
        inputs = [self.relu19(input) for input in inputs]

        # tenth block
        inputs = self.depthwise10(inputs)
        inputs = self.batch_norm20(inputs)
        inputs = [self.relu20(input) for input in inputs]
        inputs = self.pointwise_conv10(inputs)
        inputs = self.batch_norm21(inputs)
        inputs = [self.relu21(input) for input in inputs]

        # eleventh block
        inputs = self.depthwise11(inputs)
        inputs = self.batch_norm22(inputs)
        inputs = [self.relu22(input) for input in inputs]
        inputs = self.pointwise_conv11(inputs)
        inputs = self.batch_norm23(inputs)
        inputs = [self.relu23(input) for input in inputs]

        # twelfth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise12(inputs)
        inputs = self.batch_norm24(inputs)
        inputs = [self.relu24(input) for input in inputs]
        inputs = self.pointwise_conv12(inputs)
        inputs = self.batch_norm25(inputs)
        inputs = [self.relu25(input) for input in inputs]

        # thirteenth block
        inputs = self.depthwise13(inputs)
        inputs = self.batch_norm26(inputs)
        inputs = [self.relu26(input) for input in inputs]
        inputs = self.pointwise_conv13(inputs)
        inputs = self.batch_norm27(inputs)
        inputs = [self.relu27(input) for input in inputs]

        # classification block
        inputs = [tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(input) for input in inputs]
        if training:
            inputs = [tf.keras.layers.Dropout(1e-3)(input) for input in inputs]
        inputs = [tf.keras.layers.Flatten()(input) for input in inputs]
        inputs = self.head_classification(inputs)

        return inputs

    def set_subnetwork_indexes(self, subnetworks_filters_first_convolution, subnetworks_filters_depthwise,
                               subnetworks_filters_pointwise):

        for subnetwork_index in range(self.subnetworks_number - 1):

            # add filters in standard convolution layer
            self.standard_convolution.add_splitting_filters_indexes(
                subnetworks_filters_first_convolution[subnetwork_index][0] + 1)

            # add filters in depthwise and pointwise convolution layers
            pointwise_filters_subnetwork = subnetworks_filters_pointwise[subnetwork_index][0]
            depthwise_filters_subnetwork = subnetworks_filters_depthwise[subnetwork_index][0]

            block_index_depthwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_DepthwiseConv2D):
                    layer.add_splitting_filters_indexes(depthwise_filters_subnetwork[block_index_depthwise] + 1)
                    block_index_depthwise += 1

            first_layer = True

            block_index_pointwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_2DConvolution_Standard):

                    if first_layer is False:
                        layer.add_splitting_filters_indexes(pointwise_filters_subnetwork[block_index_pointwise] + 1)
                        block_index_pointwise += 1
                    else:  # skip the first standard convolution layer
                        first_layer = False

    def get_model(self):

        model = tf.keras.Sequential()
        input = tf.ones(shape=(1, 224, 224, 3))

        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase):
                batch_norm_layer = tf.keras.layers.BatchNormalization()
                batch_norm_layer(tf.squeeze(tf.ones(input.shape)))
                batch_norm_layer.set_weights(weights=layer.get_weights())
                model.add(batch_norm_layer)

            if isinstance(layer, Reds_2DConvolution_Standard):
                conv_layer = tf.keras.layers.Conv2D(filters=layer.filters, use_bias=layer.use_bias,
                                                    kernel_size=layer.kernel_size,
                                                    padding=layer.padding,
                                                    strides=layer.strides, input_shape=input.shape)
                input = conv_layer(input)
                conv_layer.set_weights(layer.get_weights()[:1])
                model.add(conv_layer)

            if isinstance(layer, Reds_DepthwiseConv2D):
                depthwise_conv_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=layer.kernel_size,
                                                                       padding=layer.padding,
                                                                       strides=layer.strides,
                                                                       use_bias=layer.use_bias)
                input = depthwise_conv_layer(input)
                depthwise_conv_layer.set_weights(layer.get_weights()[:1])
                model.add(depthwise_conv_layer)

            if isinstance(layer, tf.keras.layers.ReLU):
                model.add(tf.keras.layers.ReLU(max_value=6.0))

        return model


class Reds_MobilenetV1(tf.keras.Model):
    def __init__(self, classes, subnetworks_number=4,
                 batch_dimensions=4,
                 use_bias=False,
                 debug=False):
        super(Reds_MobilenetV1, self).__init__()

        self.subnetworks_number = subnetworks_number
        self.classes = classes

        # first standard convolution
        self.standard_convolution = Reds_2DConvolution_Standard(in_channels=3, out_channels=32,
                                                                kernel_size=(3, 3),
                                                                batch_dimensions=batch_dimensions,
                                                                use_bias=use_bias,
                                                                strides=(2,
                                                                         2),
                                                                debug=debug, padding='same')
        self.batch_norm1 = Reds_BatchNormalizationBase(fused=False)
        self.relu1 = tf.keras.layers.ReLU(max_value=6.0)

        # first block
        self.depthwise1 = Reds_DepthwiseConv2D(in_channels=32, kernel_size=(3, 3),
                                               use_bias=use_bias,
                                               strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug, padding='same')
        self.batch_norm2 = Reds_BatchNormalizationBase(fused=False)
        self.relu2 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv1 = Reds_2DConvolution_Standard(in_channels=32, out_channels=64,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm3 = Reds_BatchNormalizationBase(fused=False)
        self.relu3 = tf.keras.layers.ReLU(max_value=6.0)

        # second block
        self.depthwise2 = Reds_DepthwiseConv2D(in_channels=64, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm4 = Reds_BatchNormalizationBase(fused=False)
        self.relu4 = tf.keras.layers.ReLU(max_value=6.0)

        self.pointwise_conv2 = Reds_2DConvolution_Standard(in_channels=64, out_channels=128,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm5 = Reds_BatchNormalizationBase(fused=False)
        self.relu5 = tf.keras.layers.ReLU(max_value=6.0)

        # third block
        self.depthwise3 = Reds_DepthwiseConv2D(in_channels=128, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm6 = Reds_BatchNormalizationBase(fused=False)
        self.relu6 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv3 = Reds_2DConvolution_Standard(in_channels=128, out_channels=128,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm7 = Reds_BatchNormalizationBase(fused=False)
        self.relu7 = tf.keras.layers.ReLU(max_value=6.0)

        # fourth block
        self.depthwise4 = Reds_DepthwiseConv2D(in_channels=128, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm8 = Reds_BatchNormalizationBase(fused=False)
        self.relu8 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv4 = Reds_2DConvolution_Standard(in_channels=128, out_channels=256,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm9 = Reds_BatchNormalizationBase(fused=False)
        self.relu9 = tf.keras.layers.ReLU(max_value=6.0)

        # fifth block
        self.depthwise5 = Reds_DepthwiseConv2D(in_channels=256, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm10 = Reds_BatchNormalizationBase(fused=False)
        self.relu10 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv5 = Reds_2DConvolution_Standard(in_channels=256, out_channels=256,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm11 = Reds_BatchNormalizationBase(fused=False)
        self.relu11 = tf.keras.layers.ReLU(max_value=6.0)

        # sixth block
        self.depthwise6 = Reds_DepthwiseConv2D(in_channels=256, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(2, 2),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm12 = Reds_BatchNormalizationBase(fused=False)
        self.relu12 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv6 = Reds_2DConvolution_Standard(in_channels=256, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm13 = Reds_BatchNormalizationBase(fused=False)
        self.relu13 = tf.keras.layers.ReLU(max_value=6.0)

        # seventh block
        self.depthwise7 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm14 = Reds_BatchNormalizationBase(fused=False)
        self.relu14 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv7 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm15 = Reds_BatchNormalizationBase(fused=False)
        self.relu15 = tf.keras.layers.ReLU(max_value=6.0)

        # eighth block
        self.depthwise8 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm16 = Reds_BatchNormalizationBase(fused=False)
        self.relu16 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv8 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm17 = Reds_BatchNormalizationBase(fused=False)
        self.relu17 = tf.keras.layers.ReLU(max_value=6.0)

        # ninth block
        self.depthwise9 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                               use_bias=use_bias, strides=(1, 1),
                                               batch_dimensions=batch_dimensions,
                                               debug=debug)
        self.batch_norm18 = Reds_BatchNormalizationBase(fused=False)
        self.relu18 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv9 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                           kernel_size=(1, 1),
                                                           batch_dimensions=batch_dimensions,
                                                           use_bias=use_bias, strides=(1, 1),
                                                           debug=debug, padding='same')
        self.batch_norm19 = Reds_BatchNormalizationBase(fused=False)
        self.relu19 = tf.keras.layers.ReLU(max_value=6.0)

        # tenth block
        self.depthwise10 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm20 = Reds_BatchNormalizationBase(fused=False)
        self.relu20 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv10 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm21 = Reds_BatchNormalizationBase(fused=False)
        self.relu21 = tf.keras.layers.ReLU(max_value=6.0)

        # eleventh block
        self.depthwise11 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm22 = Reds_BatchNormalizationBase(fused=False)
        self.relu22 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv11 = Reds_2DConvolution_Standard(in_channels=512, out_channels=512,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm23 = Reds_BatchNormalizationBase(fused=False)
        self.relu23 = tf.keras.layers.ReLU(max_value=6.0)

        # twelfth block
        self.depthwise12 = Reds_DepthwiseConv2D(in_channels=512, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(2, 2),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm24 = Reds_BatchNormalizationBase(fused=False)
        self.relu24 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv12 = Reds_2DConvolution_Standard(in_channels=512, out_channels=1024,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm25 = Reds_BatchNormalizationBase(fused=False)
        self.relu25 = tf.keras.layers.ReLU(max_value=6.0)

        # thirteenth block
        self.depthwise13 = Reds_DepthwiseConv2D(in_channels=1024, kernel_size=(3, 3),
                                                use_bias=use_bias, strides=(1, 1),
                                                batch_dimensions=batch_dimensions,
                                                debug=debug)
        self.batch_norm26 = Reds_BatchNormalizationBase(fused=False)
        self.relu26 = tf.keras.layers.ReLU(max_value=6.0)
        self.pointwise_conv13 = Reds_2DConvolution_Standard(in_channels=1024, out_channels=1024,
                                                            kernel_size=(1, 1),
                                                            batch_dimensions=batch_dimensions,
                                                            use_bias=use_bias, strides=(1, 1),
                                                            debug=debug, padding='same')
        self.batch_norm27 = Reds_BatchNormalizationBase(fused=False)
        self.relu27 = tf.keras.layers.ReLU(max_value=6.0)

        self.head_classification = Linear_Adaptive(in_features=1024, out_features=classes,
                                                   debug=debug, use_bias=True)
        self.dense_model_trainable_weights = 0

    def finetune_batch_normalization(self):
        """
        Used to finetune the Batch Normalization layers while freezing all
        the other layers
        """
        for layer in self.layers:
            if isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = True
            else:
                layer.trainable = False

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def get_subnetwork_parameters_percentage(self, subnetwork_index):
        """
        If the input feature are not registered perform a forward pass on the adaptive layers to compute them, return the
        number of trainable parameters used by the subnetwork
        @param subnetwork_index:
        @return:
        """

        if self.dense_model_trainable_weights == 0:
            self.dense_model_trainable_weights = sum(
                [np.prod(tensor.shape) if not isinstance(tensor, Reds_BatchNormalizationBase) else 0 for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers[:-1]:

            if isinstance(layer, Reds_BatchNormalizationBase) or isinstance(layer,
                                                                            tf.keras.layers.Reshape) or isinstance(
                layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.LeakyReLU) or isinstance(layer,
                                                                                                           tf.keras.layers.Flatten) or isinstance(
                layer,
                tf.keras.layers.AveragePooling2D):
                continue
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def get_model_name(self):
        return "mobilenet_v1_vww"

    def compute_inference_estimations(self, input_shape=(1, 96, 96, 3)):

        layers_filters_macs = []
        layers_filters_byte = []
        layers_filters_activation_map_byte = []

        inputs = tf.ones(input_shape, dtype=tf.dtypes.float32)

        for layer in self.layers[:len(self.layers) - 1]:

            if isinstance(layer, Reds_DepthwiseConv2D) or isinstance(layer, Reds_2DConvolution_Standard):
                inputs, macs, filters_byte_memory, filters_activation_maps_memory = layer.compute_layer_inference_estimations(
                    inputs=inputs)
                layers_filters_macs.append(macs)
                layers_filters_byte.append(filters_byte_memory)
                layers_filters_activation_map_byte.append(filters_activation_maps_memory)

        return layers_filters_macs, layers_filters_byte, layers_filters_activation_map_byte

    def build(self, input_shape):
        """
        Initialize model's layers weights and biases
        """
        self.set_subnetworks_number(subnetworks_number=1)
        init_input = tf.ones(
            input_shape,
            dtype=tf.dtypes.float32,
        )
        init_input = [init_input for _ in range(1)]

        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase):
                _ = layer.build(init_input[0].shape)
            elif isinstance(layer, Reds_2DConvolution_Standard):
                init_input = layer(init_input)
            elif isinstance(layer, Reds_DepthwiseConv2D):
                init_input = layer(init_input)
            elif isinstance(layer, Linear_Adaptive):
                init_input = [tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(input) for input in init_input]
                init_input = [tf.keras.layers.Flatten()(input) for input in init_input]
                init_input = layer(init_input)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs = [inputs for _ in range(self.subnetworks_number)]

        # first standard convolution
        inputs = self.standard_convolution(inputs)
        inputs = self.batch_norm1(inputs)
        inputs = [self.relu1(input) for input in inputs]

        # first block
        inputs = self.depthwise1(inputs)
        inputs = self.batch_norm2(inputs)
        inputs = [self.relu2(input) for input in inputs]
        inputs = self.pointwise_conv1(inputs)
        inputs = self.batch_norm3(inputs)
        inputs = [self.relu3(input) for input in inputs]

        # second block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise2(inputs)
        inputs = self.batch_norm4(inputs)
        inputs = [self.relu4(input) for input in inputs]
        inputs = self.pointwise_conv2(inputs)
        inputs = self.batch_norm5(inputs)
        inputs = [self.relu5(input) for input in inputs]

        # third block
        inputs = self.depthwise3(inputs)
        inputs = self.batch_norm6(inputs)
        inputs = [self.relu6(input) for input in inputs]
        inputs = self.pointwise_conv3(inputs)
        inputs = self.batch_norm7(inputs)
        inputs = [self.relu7(input) for input in inputs]

        # fourth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise4(inputs)
        inputs = self.batch_norm8(inputs)
        inputs = [self.relu8(input) for input in inputs]
        inputs = self.pointwise_conv4(inputs)
        inputs = self.batch_norm9(inputs)
        inputs = [self.relu9(input) for input in inputs]

        # fifth block
        inputs = self.depthwise5(inputs)
        inputs = self.batch_norm10(inputs)
        inputs = [self.relu10(input) for input in inputs]
        inputs = self.pointwise_conv5(inputs)
        inputs = self.batch_norm11(inputs)
        inputs = [self.relu11(input) for input in inputs]

        # sixth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise6(inputs)
        inputs = self.batch_norm12(inputs)
        inputs = [self.relu12(input) for input in inputs]
        inputs = self.pointwise_conv6(inputs)
        inputs = self.batch_norm13(inputs)
        inputs = [self.relu13(input) for input in inputs]

        # seventh block
        inputs = self.depthwise7(inputs)
        inputs = self.batch_norm14(inputs)
        inputs = [self.relu14(input) for input in inputs]
        inputs = self.pointwise_conv7(inputs)
        inputs = self.batch_norm15(inputs)
        inputs = [self.relu15(input) for input in inputs]

        # eighth block
        inputs = self.depthwise8(inputs)
        inputs = self.batch_norm16(inputs)
        inputs = [self.relu16(input) for input in inputs]
        inputs = self.pointwise_conv8(inputs)
        inputs = self.batch_norm17(inputs)
        inputs = [self.relu17(input) for input in inputs]

        # ninth block
        inputs = self.depthwise9(inputs)
        inputs = self.batch_norm18(inputs)
        inputs = [self.relu18(input) for input in inputs]
        inputs = self.pointwise_conv9(inputs)
        inputs = self.batch_norm19(inputs)
        inputs = [self.relu19(input) for input in inputs]

        # tenth block
        inputs = self.depthwise10(inputs)
        inputs = self.batch_norm20(inputs)
        inputs = [self.relu20(input) for input in inputs]
        inputs = self.pointwise_conv10(inputs)
        inputs = self.batch_norm21(inputs)
        inputs = [self.relu21(input) for input in inputs]

        # eleventh block
        inputs = self.depthwise11(inputs)
        inputs = self.batch_norm22(inputs)
        inputs = [self.relu22(input) for input in inputs]
        inputs = self.pointwise_conv11(inputs)
        inputs = self.batch_norm23(inputs)
        inputs = [self.relu23(input) for input in inputs]

        # twelfth block
        inputs = [tf.keras.layers.ZeroPadding2D(
            ((0, 1), (0, 1))
        )(input) for input in inputs]
        inputs = self.depthwise12(inputs)
        inputs = self.batch_norm24(inputs)
        inputs = [self.relu24(input) for input in inputs]
        inputs = self.pointwise_conv12(inputs)
        inputs = self.batch_norm25(inputs)
        inputs = [self.relu25(input) for input in inputs]

        # thirteenth block
        inputs = self.depthwise13(inputs)
        inputs = self.batch_norm26(inputs)
        inputs = [self.relu26(input) for input in inputs]
        inputs = self.pointwise_conv13(inputs)
        inputs = self.batch_norm27(inputs)
        inputs = [self.relu27(input) for input in inputs]

        # classification block
        inputs = [tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(input) for input in inputs]
        if training:
            inputs = [tf.keras.layers.Dropout(1e-3)(input) for input in inputs]
        inputs = [tf.keras.layers.Flatten()(input) for input in inputs]
        inputs = self.head_classification(inputs)

        return inputs

    def set_subnetwork_indexes(self, subnetworks_filters_first_convolution, subnetworks_filters_depthwise,
                               subnetworks_filters_pointwise):

        for subnetwork_index in range(self.subnetworks_number - 1):

            # add filters in standard convolution layer
            self.standard_convolution.add_splitting_filters_indexes(
                subnetworks_filters_first_convolution[subnetwork_index][0] + 1)

            # add filters in depthwise and pointwise convolution layers
            pointwise_filters_subnetwork = subnetworks_filters_pointwise[subnetwork_index][0]
            depthwise_filters_subnetwork = subnetworks_filters_depthwise[subnetwork_index][0]

            block_index_depthwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_DepthwiseConv2D):
                    layer.add_splitting_filters_indexes(depthwise_filters_subnetwork[block_index_depthwise] + 1)
                    block_index_depthwise += 1

            first_layer = True

            block_index_pointwise = 0
            for layer in self.layers:
                if isinstance(layer, Reds_2DConvolution_Standard):

                    if first_layer is False:
                        layer.add_splitting_filters_indexes(pointwise_filters_subnetwork[block_index_pointwise] + 1)
                        block_index_pointwise += 1
                    else:  # skip the first standard convolution layer
                        first_layer = False

    def get_model(self):

        model = tf.keras.Sequential()
        input = tf.ones(shape=(1, 224, 224, 3))

        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase):
                batch_norm_layer = tf.keras.layers.BatchNormalization()
                batch_norm_layer(tf.squeeze(tf.ones(input.shape)))
                batch_norm_layer.set_weights(weights=layer.get_weights())
                model.add(batch_norm_layer)

            if isinstance(layer, Reds_2DConvolution_Standard):
                conv_layer = tf.keras.layers.Conv2D(filters=layer.filters, use_bias=layer.use_bias,
                                                    kernel_size=layer.kernel_size,
                                                    padding=layer.padding,
                                                    strides=layer.strides, input_shape=input.shape)
                input = conv_layer(input)
                conv_layer.set_weights(layer.get_weights()[:1])
                model.add(conv_layer)

            if isinstance(layer, Reds_DepthwiseConv2D):
                depthwise_conv_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=layer.kernel_size,
                                                                       padding=layer.padding,
                                                                       strides=layer.strides,
                                                                       use_bias=layer.use_bias)
                input = depthwise_conv_layer(input)
                depthwise_conv_layer.set_weights(layer.get_weights()[:1])
                model.add(depthwise_conv_layer)

            if isinstance(layer, tf.keras.layers.ReLU):
                model.add(tf.keras.layers.ReLU(max_value=6.0))

        return model


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MobileNet v1 models for Keras.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
-------------------------|---------------|-------------------|--------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
----------------------|---------------|-------------------|----------------
|  1.0 MobileNet-224  |    70.6 %    |        569        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        418        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        290        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        186        |     4.2     |

Reference:
  - [MobileNets: Efficient Convolutional Neural Networks
     for Mobile Vision Applications](
      https://arxiv.org/abs/1704.04861)
"""

from keras import backend
from keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers

from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/"
)
layers = None


@keras_export(
    "keras.applications.mobilenet.MobileNet", "keras.applications.MobileNet"
)
def Reds_MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=1e-3,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        **kwargs,
):
    """Instantiates the MobileNet architecture.

    Reference:
    - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNet, call `tf.keras.applications.mobilenet.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet.preprocess_input` will scale input pixels between -1 and 1.

    Args:
      input_shape: Optional shape tuple, only to be specified if `include_top`
        is False (otherwise the input shape has to be `(224, 224, 3)` (with
        `channels_last` data format) or (3, 224, 224) (with `channels_first`
        data format). It should have exactly 3 inputs channels, and width and
        height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
        valid value. Default to `None`.
        `input_shape` will be ignored if the `input_tensor` is provided.
      alpha: Controls the width of the network. This is known as the width
        multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
        decreases the number of filters in each layer. - If `alpha` > 1.0,
        proportionally increases the number of filters in each layer. - If
        `alpha` = 1, default number of filters from the paper are used at each
        layer. Default to 1.0.
      depth_multiplier: Depth multiplier for depthwise convolution. This is
        called the resolution multiplier in the MobileNet paper. Default to 1.0.
      dropout: Dropout rate. Default to 0.001.
      include_top: Boolean, whether to include the fully-connected layer at the
        top of the network. Default to `True`.
      weights: One of `None` (random initialization), 'imagenet' (pre-training
        on ImageNet), or the path to the weights file to be loaded. Default to
        `imagenet`.
      input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model. `input_tensor` is useful for sharing
        inputs between multiple different networks. Default to None.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` (default) means that the output of the model will be
            the 4D tensor output of the last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: Optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified. Defaults to 1000.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """
    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError(f"Unknown argument(s): {(kwargs,)}")
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received classes={classes}"
        )

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if depth_multiplier != 1:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "depth multiplier must be 1.  "
                f"Received depth_multiplier={depth_multiplier}"
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of"
                "`0.25`, `0.50`, `0.75` or `1.0` only.  "
                f"Received alpha={alpha}"
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    imgs_input = [tf.ones(shape=(1, 224, 224, 3)) for _ in range(1)]
    x = _conv_block(imgs_input, 32, alpha, strides=(2, 2))

    # first block
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, in_channels=32)

    # second block
    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), in_channels=64
    )

    # third block
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, in_channels=128)

    # fourth block
    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), in_channels=128
    )
    # fifth block
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, in_channels=256)

    # sixth block
    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), in_channels=256
    )
    # seventh block
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, in_channels=512)

    # eighth block
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, in_channels=512)

    # ninth block
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, in_channels=512)

    # tenth block
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, in_channels=512)

    # eleventh block
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, in_channels=512)

    # twelfth block
    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), in_channels=512
    )

    # thirteenth block
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, in_channels=1024)

    if include_top:
        x = Reds_GlobalAveragePooling2D(keepdims=True)(x)
        # x = layers.Dropout(dropout, name="dropout")(x)
        x = Reds_2DConvolution_Standard(in_channels=1024, out_channels=classes, kernel_size=(1, 1), padding="same")(x)
        x = Reds_Reshape((classes,), name="reshape_2")(x[0])
        imagenet_utils.validate_activation(classifier_activation, weights)
        # x = Reds_Activation(
        #    activation=classifier_activation, name="predictions"
        # )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x[0], name=f"mobilenet_{alpha:0.2f}_{rows}")

    """
    
    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    """
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = int(filters * alpha)
    x, times = Reds_2DConvolution_Standard(
        in_channels=3,
        out_channels=filters,
        kernel_size=kernel,
        padding="same",
        use_bias=False,
        strides=strides,
    )(inputs)
    x = Reds_BatchNormalizationBase(axis=channel_axis, fused=False)(x)
    return Reds_ReLU(6.0)(x)


def _depthwise_conv_block(
        inputs,
        pointwise_conv_filters,
        alpha,
        depth_multiplier=1,
        strides=(1, 1),
        in_channels=None
):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = Reds_ZeroPadding2D(
            ((0, 1), (0, 1))
        )(inputs)
    x = Reds_DepthwiseConv2D(
        in_channels=in_channels,
        kernel_size=(3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False
    )(x)
    x = Reds_BatchNormalizationBase(
        axis=channel_axis, fused=False
    )(x)
    x = Reds_ReLU(6.0)(x)

    x, times = Reds_2DConvolution_Standard(
        in_channels=in_channels,
        out_channels=pointwise_conv_filters,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1)
    )(x)
    x = Reds_BatchNormalizationBase(
        axis=channel_axis, fused=False
    )(x)
    return Reds_ReLU(6.0)(x)


@keras_export("keras.applications.mobilenet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


@keras_export("keras.applications.mobilenet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)
