import os
import sys

import scipy.ndimage as nd
import tensorflow as tf
import numpy as np
import cv2

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def getBestShift(img):
    cy,cx = nd.measurements.center_of_mass(img)
    print (cy,cx)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

lable = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28=784

y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes
W1 = tf.get_variable("W1", shape=[784, 500], initializer=xaver_init(784, 500))
W2 = tf.get_variable("W2", shape=[500, 256], initializer=xaver_init(784, 256))
W3 = tf.get_variable("W3", shape=[256, 10], initializer=xaver_init(500, 256))

b1 = tf.Variable(tf.zeros([500]))
b2 = tf.Variable(tf.zeros([256]))

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))  # Softmax
b3 = tf.Variable(tf.zeros([10]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))  # Softmax
activation = tf.add(tf.matmul(L2, W3), b3)  # Softmax


cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))  # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # Gradient Descent

image = sys.argv[1]

if not os.path.exists(image):
    print ("File " + image + " doesn't exist")
    exit(1)

color_complete = cv2.imread(image)
gray_complete = cv2.imread(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
(thresh, gray_complete) = cv2.threshold(255 - gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imwrite("pro-img/compl.png", gray_complete)

digit_image = -np.ones(gray_complete.shape)
height, width = gray_complete.shape


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

checkpoint_dir = "cps/"

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    print 'load learning'
    saver.restore(sess, ckpt.model_checkpoint_path)

else:
    print 'fail load learning'
    print 'you have to run traning.py'
    exit(1)

for cropped_width in range(50, 1000, 20):
    for cropped_height in range(50, 1000, 20):
        for shift_x in range(0, width - cropped_width, cropped_width / 4):
            for shift_y in range(0, height - cropped_height, cropped_height /4):
                gray = gray_complete[shift_y:shift_y + cropped_height, shift_x:shift_x + cropped_width]
                if np.count_nonzero(gray) <= 20:
                    continue

                if (np.sum(gray[0]) != 0) or (np.sum(gray[:, 0]) != 0) or (np.sum(gray[-1]) != 0) or (
                            np.sum(gray[:, -1]) != 0):
                    continue

                top_left = np.array([shift_y, shift_x])
                bottom_right = np.array([shift_y + cropped_height, shift_x + cropped_width])

                while np.sum(gray[0]) == 0:
                    top_left[0] += 1
                    gray = gray[1:]

                while np.sum(gray[:, 0]) == 0:
                    top_left[1] += 1
                    gray = np.delete(gray, 0, 1)

                while np.sum(gray[-1]) == 0:
                    bottom_right[0] -= 1
                    gray = gray[:-1]

                while np.sum(gray[:, -1]) == 0:
                    bottom_right[1] -= 1
                    gray = np.delete(gray, -1, 1)

                actual_w_h = bottom_right - top_left
                if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] + 1) >
                                0.2 * actual_w_h[0] * actual_w_h[1]):
                    continue

                print ("------------------")
                print ("------------------")

                rows, cols = gray.shape
                compl_dif = abs(rows - cols)
                half_Sm = compl_dif / 2
                half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
                if rows > cols:
                    gray = np.lib.pad(gray, ((0, 0), (half_Sm, half_Big)), 'constant')
                else:
                    gray = np.lib.pad(gray, ((half_Sm, half_Big), (0, 0)), 'constant')

                gray = cv2.resize(gray, (20, 20))
                gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')

                shiftx, shifty = getBestShift(gray)
                shifted = shift(gray, shiftx, shifty)
                gray = shifted

                """
                and not from 0-255 so we divide our flatten images
                all images in the training set have an range from 0-1
                (a one dimensional vector with our 784 pixels)
                to use the same 0-1 based range
                """

                feed = gray.flatten() / 255.0

                print "Prediction for ", (shift_x, shift_y, cropped_width)
                print "Pos"
                print top_left
                print bottom_right
                print actual_w_h
                print " "


                prediction = [tf.reduce_max(activation), tf.argmax(activation, 1)[0]]
                pred = sess.run(prediction, feed_dict={x: [feed]})
                print 'the number is ' + str(pred[1])
                print str(pred[0] * 100) + '%'

                cv2.imwrite("pro-img/" + str(shift_x) + "_" + str(shift_y) + " : " + str(pred[1]) + "_" + str(pred[0] * 100)+"%.png", gray)

                check = input("is it right? : ")

                if (check == True):
                    sess.run(optimizer, feed_dict={x: [feed], y: [lable[pred[1]]]})

                elif(check == False):
                    num = input("what is that? : ")
                    sess.run(optimizer, feed_dict={x: [feed], y: [lable[num]]})



                digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = pred[1]

                cv2.rectangle(color_complete, tuple(top_left[::-1]), tuple(bottom_right[::-1]), color=(0, 255, 0),
                              thickness=5)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_complete, str(pred[1]), (top_left[1], bottom_right[0] + 50),
                            font, fontScale=1, color=(0, 255, 0), thickness=4)
                cv2.putText(color_complete, format(pred[0] * 100, ".1f") + "%",
                            (top_left[1] + 30, bottom_right[0] + 60),
                            font, fontScale=0.5, color=(0, 255, 0), thickness=2)

cv2.imwrite("pro-img/digitized_image.png", color_complete)
saver.save(sess, checkpoint_dir + 'model.ckpt')
print 'finish the test'

