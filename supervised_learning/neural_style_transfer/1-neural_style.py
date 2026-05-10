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
