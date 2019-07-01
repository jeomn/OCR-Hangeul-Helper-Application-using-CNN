import tensorflow as tf


#가중치 값
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


#편향 값
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')


#모델 정의
#1 Convolution 계층
def Conv1(input_data):
    with tf.name_scope('conv_1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1_relu = tf.nn.relu(h_conv1 + b_conv1)
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv1_maxpool

#2 Convolution 계층
def Conv2(input_data):
    with tf.name_scope('conv_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.conv2d(input_data, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2_relu = tf.nn.relu(h_conv2 + b_conv2)
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv2_maxpool

#3 Convolution 계층
def Conv3(input_data):
    with tf.name_scope('conv_3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.conv2d(input_data, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3_relu = tf.nn.relu(h_conv3 + b_conv3)
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_conv3_maxpool

#Fully Connected 계층
def ful_Con(input_data):
    with tf.name_scope('ful_con'):
        h_pool_flat = tf.reshape(input_data, [-1, 8 * 8 * 128])
        W_fc1 = weight_variable([8 * 8 * 128, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        return h_fc1

class Model:
    #CNN network
    def CNN_model(image, keep_prob, num_label, output_node_name):
        cnn_conv1 = Conv1(image)
        cnn_conv2 = Conv2(cnn_conv1)
        cnn_conv3 = Conv3(cnn_conv2)
        cnn_fc1 = ful_Con(cnn_conv3)

        #Drop out 계층
        h_fc1_drop = tf.nn.dropout(cnn_fc1, keep_prob)

        # 분류 계층
        W_fc2 = weight_variable([1024, num_label])
        b_fc2 = bias_variable([num_label])
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        y_pred = tf.nn.softmax(y, name=output_node_name)

        return y, y_pred