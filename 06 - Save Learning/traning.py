import tensorflow as tf

import input_data



learning_rate = 0.01
training_epochs = 2000
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))  # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
checkpoint_dir = "cps/"

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print 'load learning'
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (sess.run(b))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    saver.save(sess, checkpoint_dir + 'model.ckpt')

