import tensorflow as tf

gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus))
