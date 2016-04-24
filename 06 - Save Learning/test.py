import os
import sys


import scipy.ndimage as nd
import tensorflow as tf
import numpy as np
import cv2




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


x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10])  # 0-9 digits recognition => 10 classes
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

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

for cropped_width in range(100, 300, 20):
    for cropped_height in range(100, 300, 20):
        for shift_x in range(0, width - cropped_width, cropped_width / 4):
            for shift_y in range(0, height - cropped_height, cropped_height / 4):
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

                cv2.imwrite("pro-img/" + str(shift_x) + "_" + str(shift_y) + ".png", gray)

                """
                and not from 0-255 so we divide our flatten images
                all images in the training set have an range from 0-1
                (a one dimensional vector with our 784 pixels)
                to use the same 0-1 based range
                """

                feed = gray.flatten()
                feed = np.reshape(feed, (-1, 784))

                print "Prediction for ", (shift_x, shift_y, cropped_width)
                print "Pos"
                print top_left
                print bottom_right
                print actual_w_h
                print " "

                pred = sess.run(tf.argmax(activation, 1), feed_dict={x: feed})
                print 'the number is ' + str(pred)

                digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = pred[0]

                cv2.rectangle(color_complete, tuple(top_left[::-1]), tuple(bottom_right[::-1]), color=(0, 255, 0),
                              thickness=5)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color_complete, str(pred[0]), (top_left[1], bottom_right[0] + 50),
                            font, fontScale=1.4, color=(0, 255, 0), thickness=4)
                cv2.putText(color_complete, format(pred[0] * 100, ".1f") + "%",
                            (top_left[1] + 30, bottom_right[0] + 60),
                            font, fontScale=0.8, color=(0, 255, 0), thickness=2)

cv2.imwrite("pro-img/digitized_image.png", color_complete)
