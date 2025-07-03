import tensorflow as tf


def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # for now random weights; later load real ones
    return model
