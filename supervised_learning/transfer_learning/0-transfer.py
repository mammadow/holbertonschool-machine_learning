#!/usr/bin/env python3
"""Transfer learning for CIFAR-10 using MobileNetV2."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for MobileNetV2."""
    X_p = K.applications.mobilenet_v2.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


@K.utils.register_keras_serializable()
def resize_images(x):
    """Resize CIFAR-10 images from 32x32 to 96x96."""
    return K.ops.image.resize(x, (96, 96))


def build_model():
    """Build the transfer learning model."""
    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(
        resize_images,
        output_shape=(96, 96, 3)
    )(inputs)
    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    """Train and save the model."""
    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)

    model = build_model()
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p,
        Y_train_p,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )

    model.save('cifar10.h5')


if __name__ == '__main__':
    main()
