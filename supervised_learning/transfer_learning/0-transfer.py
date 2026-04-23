#!/usr/bin/env python3
"""Train a transfer learning model on CIFAR-10."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for MobileNetV2."""
    X_p = K.applications.mobilenet_v2.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_feature_extractor():
    """Build the frozen MobileNetV2 feature extractor."""
    inputs = K.Input(shape=(32, 32, 3))

    resized = K.layers.Lambda(
            lambda x: K.backend.resize_images(
                x, 3, 3,
                data_format='channels_last',
                interpolation='bilinear'
            )
    )(inputs)

    base_model = K.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
    )
    base_model.trainable = False
    features = base_model(resized, training=False)
    pooled = K.layers.GlobalAveragePooling2D()(features)
    model = K.Model(inputs=inputs, outputs=pooled)
    return model, base_model


def build_classifier(input_dim):
    """Build the classifier head trained on cached bottleneck features."""
    inputs = K.Input(shape=(input_dim,))
    x = K.layers.Dense(512, activation='relu')(inputs)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    return K.Model(inputs=inputs, outputs=outputs)


def build_full_model(feature_extractor, classifier):
    """Attach the trained classifier head to the feature extractor."""
    inputs = feature_extractor.input
    outputs = classifier(feature_extractor.output)
    return K.Model(inputs=inputs, outputs=outputs)


def unfreeze_top_layers(base_model, num_layers=40):
    """Unfreeze the last layers of the base model except BN layers."""
    for layer in base_model.layers[-num_layers:]:
        if not isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = True


def main():
    """Train, fine-tune, and save the CIFAR-10 model."""
    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)

    X_val_p = X_train_p[-5000:]
    Y_val_p = Y_train_p[-5000:]
    X_train_p = X_train_p[:-5000]
    Y_train_p = Y_train_p[:-5000]

    feature_extractor, base_model = build_feature_extractor()

    train_features = feature_extractor.predict(
            X_train_p, batch_size=128, verbose=1
    )
    val_features = feature_extractor.predict(
            X_val_p, batch_size=128, verbose=1
    )
    classifier = build_classifier(train_features.shape[1])
    classifier.compile(
            optimizer = K.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
    )
    callbacks = [
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
    ]
    
    classifier.fit(
            train_features,
            Y_train_p,
            validation_data=(val_features, Y_val_p),
            epochs=25,
            batch_size=128,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
    )

    model = build_full_model(feature_extractor, classifier)
    unfreeze_top_layers(base_model, 40)

    model.compile(
            optimizer=K.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
    )

    datagen = K.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest'
    )
    datagen.fit(X_train_p)

    fine_tune_callbacks = [
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-7
            )
    ]

    model.fit(
            datagen.flow(X_train_p, Y_train_p, batch_size=128),
            validation_data=(X_val_p, Y_val_p),
            epochs=15,
            callbacks=fine_tune_callbacks,
            verbose=1
    )

    model.save('cifar10.h5')


if __name__ == '__main__':
    main()

