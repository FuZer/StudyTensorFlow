import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

print(sess.run(hello))
