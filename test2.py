import tensorflow as tf
# from readDate import *
import numpy as np
slim = tf.contrib.slim
import tensorflow as tf
import glob
import os
import scipy.io
import random
import matplotlib.pyplot as plt
import math

####model
def vgg_16(inputs, num_classes=2,is_training=True,dropout_keep_prob=0.5,scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs],reuse=tf.AUTO_REUSE) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_regularizer=slim.l2_regularizer(0.00005)
                            ):
          net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', trainable=True)
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', trainable=True)
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', trainable=True)
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', trainable=True)
          net = slim.max_pool2d(net, [2, 2], scope='pool4')
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', trainable=True)
          net = slim.max_pool2d(net, [2, 2], scope='pool5')
          # Use conv2d instead of fully_connected layers.
          net = slim.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc6', trainable=True)
          net = slim.dropout(net, dropout_keep_prob, is_training=True,
                             scope='dropout6')
          net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
          net = slim.dropout(net, dropout_keep_prob, is_training=True,
                             scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc8')
          net = tf.squeeze(net,[1,2])
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
          return net, end_points
####train
x = tf.placeholder(tf.float32, [None, 224, 224, 1] , name='input')
label_one_hot = tf.placeholder(tf.float32, [None, 2], name='label')
predict, _ = vgg_16(x)
####loss
global_step = tf.Variable(0,trainable=False)
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_one_hot, logits=predict)
# l2_loss = tf.losses.get_regularization_loss()
loss_all = tf.losses.get_total_loss()
####train
#################learning_rate_config
learning_rate = tf.train.exponential_decay(1e-6,global_step,500,0.9,staircase=True)
# learning_rates = [1e-1,1e-3,1e-5,1e-7]
# boundaries = [10000,50000,100000]
# learning_rate = tf.train.piecewise_constant(global_step,boundaries=boundaries,values=learning_rates)
###########Move_averages
# variables_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
# variables_averages_op = variables_averages.apply(tf.trainable_variables())
# train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
# train_op = tf.group([train_step, variables_averages_op])
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,global_step=global_step)

corret_pred = tf.equal(tf.arg_max(predict, 1), tf.arg_max(label_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(corret_pred, tf.float32))
tf.summary.scalar('acc',accuracy)
tf.summary.scalar('loss',loss_all)
tf.summary.scalar('lr',learning_rate)

# tf.summary.image('image',x)






model_vars = tf.model_variables()
print(model_vars)
vgg_16_vars = [var for var in model_vars if 'fc' not in var.name and 'conv1' not in var.name]
saver_vgg = tf.train.Saver(vgg_16_vars)

saver = tf.train.Saver()
model_vars = tf.trainable_variables()
print(model_vars)

all_TP = all_FP = all_TN = all_FN = 0
Label_list = []
Logits_list = []
with tf.Session() as sess:

    tf.global_variables_initializer().run()
    saver_vgg.restore(sess,
                      '/mnt/data/fast-neural-style-tensorflow-master/pretrained/vgg16.ckpt')
    writer = tf.summary.FileWriter('/mnt/data/VGG_ultrasound/record/train', sess.graph)

    writer_valid = tf.summary.FileWriter(
        '/mnt/data/VGG_ultrasound/record/val')


    saver.restore(sess,'/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/model/1/model.ckpt-16001')

## 39001 0.86
    img_path_ill = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/test/ill/1/'
    imgList_ill = os.listdir(img_path_ill)
    img_path_normal = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/test/normal/'
    imgList_normal = os.listdir(img_path_normal)
    imgList = imgList_ill+imgList_normal
    merge_writer = tf.summary.merge_all()
    i=0
    for img_name in imgList:
        for name,labelname in [['ill', np.array([0, 1])], ['normal', np.array([1, 0])]]:
######################## dataread


            img_dir = os.path.join(img_path, img_name)

            img_dir = os.path.join(img_dir, name)
            mat_list = os.listdir(img_dir)

            for mat_name in mat_list:
                img_final_name = os.path.join(img_dir, mat_name)
                print(img_dir)
                data = scipy.io.loadmat(img_final_name)
                inter_name = name + '_Array'
                data = data[inter_name]
                data = (data-data.min())/(data.max()-data.min())

                w = data.shape[1]
                h = data.shape[0]

                expend = math.ceil(224 / w)
                data = np.tile(data, (1, expend))
                data_resize = data[:, 0:224]
                Img = np.resize(data_resize, (224, 224))


                Img = np.expand_dims(Img, axis=0)
                Img = np.expand_dims(Img, axis=3)

                Label_one_hot_train = labelname
                Label_one_hot_train = np.expand_dims(Label_one_hot_train, axis=0)



                Loss,Predict = sess.run([loss_all,predict], feed_dict={x: Img, label_one_hot: Label_one_hot_train})
                print('The training step is [{0}], The loss is [{1}]'.format(i, Loss))
                # print('The training step is [{0}], The l2loss is [{1}]'.format(i, l2loss))
                print(Predict)


                logits_reshape = np.reshape(Predict, [1, -1])[0][0]
                Logits_list.append(logits_reshape)


                label_reshape = np.reshape(Label_one_hot_train, [1, -1])[0][0]
                Label_list.append(label_reshape)

                Predict = np.argmax(Predict, axis=1)
                Label_one_hot_train = np.argmax(Label_one_hot_train, axis=1)
                TP = np.sum(
                    np.logical_and(np.equal(Label_one_hot_train, 1), np.equal(Predict, 1)).astype(int)
                )
                TN = np.sum(
                    np.logical_and(np.equal(Label_one_hot_train, 0), np.equal(Predict, 0)).astype(int)
                )
                FP = np.sum(
                    np.logical_and(np.equal(Label_one_hot_train, 0), np.equal(Predict, 1)).astype(int)
                )
                FN = np.sum(
                    np.logical_and(np.equal(Label_one_hot_train, 1), np.equal(Predict, 0)).astype(int)
                )
                all_TP = all_TP + TP
                all_FP = all_FP + FP
                all_TN = all_TN + TN
                all_FN = all_FN + FN
                print('label{} ======= predict{}'.format(Label_one_hot_train, Predict))
                print('TP:{}'.format(TP))
                print('FP:{}'.format(FP))
                print('TN:{}'.format(TN))
                print('FN:{}'.format(FN))
                i=i+1
    precision = all_TP / (all_TP + all_FP)
    recall = all_TP / (all_TP + all_FN)
    sensitivity = all_TP / (all_TP + all_FN)
    specificity = all_TN / (all_TN + all_FP)
    ACC = (all_TP + all_TN) / (all_TP + all_TN + all_FP + all_FN)

    print('all_TP:{}'.format(all_TP))
    print('all_FP:{}'.format(all_FP))
    print('all_TN:{}'.format(all_TN))
    print('all_FN:{}'.format(all_FN))

    print('ACC:{}'.format(ACC))

    print('the precision is [{}]'.format(precision))
    print('the recall is [{}]'.format(recall))
    print('the sensitivity is [{}]'.format(sensitivity))
    print('the specificity is [{}]'.format(specificity))

    ###
    threshold = sorted(Logits_list, reverse=True)
    print(threshold)


    def fuc(x, i):
        if x >= i:
            x = 1
        else:
            x = 0
        return x


    y = [0]
    x = [0]

    for i in threshold:
        pred = list(map(fuc, Logits_list, [i] * len(Logits_list)))

        tp = np.logical_and(np.equal(pred, 1), np.equal(Label_list, 1))
        tp = tp.astype(np.int16).sum()
        fp = np.logical_and(np.equal(pred, 1), np.equal(Label_list, 0))
        fp = fp.astype(np.int16).sum()

        tn = np.logical_and(np.equal(pred, 0), np.equal(Label_list, 0))
        tn = tn.astype(np.int16).sum()
        fn = np.logical_and(np.equal(pred, 0), np.equal(Label_list, 1))
        fn = fn.astype(np.int16).sum()

        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        y.append(tpr)
        x.append(fpr)

    plt.figure(figsize=(10, 10))
    plt.plot(x, y, color='darkorange')
    plt.title('ROC', fontsize=20)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.savefig('/mnt/data/VGG_ultrasound/RF/spectrogram/roc.jpg')