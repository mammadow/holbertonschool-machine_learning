#!/usr/bin/env python3
"""Neural style transfer module."""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance."""
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescale an image so its largest side is 512 pixels."""
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        height = image.shape[0]
        width = image.shape[1]

        scale = 512 / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """Create the model used to calculate neural style transfer cost."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        for layer in vgg.layers:
            layer.trainable = False

        vgg_config = vgg.get_config()

        for layer in vgg_config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'

        avg_vgg = tf.keras.Model.from_config(vgg_config)
        avg_vgg.set_weights(vgg.get_weights())

        avg_vgg.trainable = False

        for layer in avg_vgg.layers:
            layer.trainable = False

        outputs = [
            avg_vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]

        self.model = tf.keras.Model(inputs=avg_vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the gram matrix of an input layer."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bhwi,bhwj->bij', input_layer, input_layer)

        shape = tf.shape(input_layer)
        height = shape[1]
        width = shape[2]

        gram = gram / tf.cast(height * width, tf.float32)

        return gram

    def generate_features(self):
        """Extract style and content features."""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.style_features = style_outputs[:len(self.style_layers)]

        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in self.style_features
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        channels = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                len(gram_target.shape) != 3 or
                gram_target.shape[0] != 1 or
                gram_target.shape[1] != channels or
                gram_target.shape[2] != channels):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(channels, channels)
            )

        gram_style = self.gram_matrix(style_output)

        cost = tf.reduce_sum(tf.square(gram_style - gram_target))
        cost = cost / tf.cast(channels ** 2, tf.float32)

        return cost

    def style_cost(self, style_outputs):
        """Calculate the style cost for the generated image."""
        length = len(self.style_layers)

        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(length)
            )

        weight = 1 / length
        cost = 0

        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            cost += weight * self.layer_style_cost(
                style_output,
                gram_target
            )

        return cost

    def content_cost(self, content_output):
        """Calculate the content cost for the generated image."""
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}"
                .format(self.content_feature.shape)
            )

        cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

        return cost

    def total_cost(self, generated_image):
        """Calculate the total cost for the generated image."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != self.content_image.shape):
            raise TypeError(
                "generated_image must be a tensor of shape {}"
                .format(self.content_image.shape)
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )

        outputs = self.model(preprocessed)

        style_outputs = outputs[:len(self.style_layers)]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style
