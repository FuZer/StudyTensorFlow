import tensorflow as tf

# data set
x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

# try to find values for w and b that compute y_data = W * x_data + b
# range is -100 ~ 100
W = tf.Variable(tf.random_uniform([1], -100., 1000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
hypothesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in xrange(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.5})
