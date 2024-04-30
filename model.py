import tensorflow as tf 


def model_load():
    model = tf.keras.models.load_model('model_over600.h5', compile=False)
    return model