import tensorflow as tf
#tf.enable_eager_execution()
#tf.executing_eagerly()
#print(tf.executing_eagerly())
import numpy as np
from array import array
import keras.backend as K
# x = np.array([[1, 2, 3], [1,2,3]], dtype='float32')
# y = np.array([[2, 3, 4], [2,3,4]], dtype='float32')
# print(x)
# print(y)

#very inspired by/credit to: https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b
def my_weighted_mse(y_pred, y_true):
    loss = K.square(y_pred - y_true)
    w = [1., 0.25465, 0.0636625] #first two values from spreadsheet, setting phi to be the same as theta as Greg suggested for a first test. randomly setting to a quarter of the weight of theta 
    print(w)
    loss = loss * w
    loss = K.mean(loss, axis=1)

    return loss

# secondOut = my_weighted_mse(x,y)
# print(secondOut)
# print(secondOut.numpy())
