import tensorflow as tf

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))
