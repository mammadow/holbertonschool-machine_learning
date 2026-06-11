#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# 1. Create the data file the checker expects
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
np.savez("MNIST.npz", X_test=x_test)

# 2. Build your model and run the checker's exact evaluation
auto = __import__('3-variational').autoencoder(784, [512], 2)[2]
x = np.load("MNIST.npz")["X_test"][:256].reshape((-1, 784)).astype('float32')

print("raw 0-255 :", auto.evaluate(x, x, verbose=False))
print("normalized:", auto.evaluate(x / 255., x / 255., verbose=False))
print("reference :", 543.7645)
