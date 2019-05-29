import tensorflow as tf
# from readDate import *
import  numpy as np
slim = tf.contrib.slim
import tensorflow as tf
import glob
import os
import scipy.io
import random
import cv2
import math
from imgaug import augmenters as iaa

#####aug
def augmentNumpy(imgs):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq_imgs = iaa.Sequential(
        [iaa.Fliplr(0.5),  # horizontally flip 50% of all images
         iaa.Flipud(0.2),
         iaa.Affine(
             scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
             rotate=(-15, 15),
             shear=(-8, 8)

         )

         ], random_order=True)
    seq_imgs_deterministic = seq_imgs.to_deterministic()

    imgs_aug = seq_imgs_deterministic.augment_image(imgs)

    return imgs_aug
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

def fuc(x, i):
    img_part_path = os.path.join(str(i), x)
    return (img_part_path)


def read_data(train_set, data_path):
    glob.glob(data_path)
    img_list = []
    for i in train_set:
        train_path = os.path.join(data_path, str(i))
        part_list = list(map(fuc, os.listdir(train_path), [i] * len(os.listdir(train_path))))
        img_list += part_list
    return img_list

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
learning_rate = tf.train.exponential_decay(1e-6,global_step,5000,0.9,staircase=True)
# learning_rate = 1e-7
# # learning_rates = [1e-1,1e-3,1e-5,1e-7]
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
mean_acc = tf.placeholder(tf.float32,name='mean_acc')
mean_loss = tf.placeholder(tf.float32,name='mean_loss')

tf.summary.scalar('acc',mean_acc)
tf.summary.scalar('loss',loss_all)
tf.summary.scalar('lr',learning_rate)

# tf.summary.image('image',x)






model_vars = tf.model_variables()
print(model_vars)
vgg_16_vars = [var for var in model_vars if 'fc' not in var.name and 'conv1' not in var.name]
saver_vgg = tf.train.Saver(vgg_16_vars)

saver = tf.train.Saver(max_to_keep=100)
model_vars = tf.trainable_variables()
print(model_vars)
with tf.Session() as sess:
    epochs = 500
    batchs = 1000
    tf.global_variables_initializer().run()
    saver_vgg.restore(sess,
                      '/mnt/data/fast-neural-style-tensorflow-master/pretrained/vgg16.ckpt')


    val_num = 1
    imgList = read_data(train_set=[2, 3, 4, 5],data_path='/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/data')
    valList = read_data(train_set=[val_num],data_path='/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/data')
    print('The train dataser is {}'.format(imgList))
    print('The val dataser is {}'.format(valList))

    writer = tf.summary.FileWriter('/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/record/{}/train'.format(str(val_num)), sess.graph)
    # saver.restore(sess,'/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/model/5/model.ckpt-2501')

    writer_valid = tf.summary.FileWriter('/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/record/{}/val'.format(str(val_num)))
    img_path ='/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/data'
    # imgList = os.listdir(img_path)
    all_validata_acc = 0

    all_training_acc = 0
    merge_writer = tf.summary.merge_all()
#### train
    for epoch in range(epochs):
        for i in range(batchs):
            img_name = random.choice(imgList)
            img_dir = os.path.join(img_path,img_name)
            seed_random = random.randint(0,1)
######################## dataread
            if seed_random == 0:
                img_dir = os.path.join(img_dir, 'ill')
                mat_list = os.listdir(img_dir)
                mat_name = random.choice(mat_list)
                img_dir = os.path.join(img_dir, mat_name)
                data = scipy.io.loadmat(img_dir)
                data = data['ill_Array']
                data = (data-data.min())/(data.max()-data.min())
                w = data.shape[1]
                h = data.shape[0]
                expend = math.ceil(224 / w)
                data = np.tile(data, (1, expend))
                data_resize = data[:, 0:224]
                Img = np.resize(data_resize, (224, 224))

                # Img = augmentNumpy(Img)

                Img = np.expand_dims(Img, axis=0)
                Img = np.expand_dims(Img, axis=3)

                Label_one_hot_train = np.array([0,1])
                Label_one_hot_train = np.expand_dims(Label_one_hot_train, axis=0)

            else:
                img_dir = os.path.join(img_dir, 'normal')
                mat_list = os.listdir(img_dir)
                mat_name = random.choice(mat_list)
                img_dir = os.path.join(img_dir, mat_name)
                data = scipy.io.loadmat(img_dir)
                data = data['normal_Array']

                data = (data-data.min())/(data.max()-data.min())
                w = data.shape[1]
                h = data.shape[0]
                expend = math.ceil(224 / w)
                data = np.tile(data, (1, expend))
                data_resize = data[:, 0:224]
                Img = np.resize(data_resize, (224, 224))
                # Img = augmentNumpy(Img)

                Img = np.expand_dims(Img, axis=0)
                Img = np.expand_dims(Img, axis=3)
                Label_one_hot_train = np.array([1,0])
                Label_one_hot_train = np.expand_dims(Label_one_hot_train, axis=0)


            sess.run(train_step, feed_dict={x: Img, label_one_hot: Label_one_hot_train})
            Loss = sess.run([loss_all], feed_dict={x: Img, label_one_hot: Label_one_hot_train})
            print('The training step is [{0}], The loss is [{1}]'.format(i, Loss))
            # print('The training step is [{0}], The l2loss is [{1}]'.format(i, l2loss))
            # if i % 500 == 0:

            img_name = random.choice(valList)
            img_dir = os.path.join(img_path, img_name)
            seed_random = random.randint(0, 1)
            ######################## dataread
            if seed_random == 0:
                img_dir = os.path.join(img_dir, 'ill')
                mat_list = os.listdir(img_dir)
                mat_name = random.choice(mat_list)
                img_dir = os.path.join(img_dir, mat_name)
                data = scipy.io.loadmat(img_dir)
                data = data['ill_Array']
                data = (data - data.min()) / (data.max() - data.min())

                w = data.shape[1]
                h = data.shape[0]

                expend = math.ceil(224 / w)
                data = np.tile(data, (1, expend))
                data_resize = data[:, 0:224]
                val_Img = np.resize(data_resize, (224, 224))

                val_Img = np.expand_dims(val_Img, axis=0)
                val_Img = np.expand_dims(val_Img, axis=3)
                Label_one_hot_val = np.array([0, 1])
                Label_one_hot_val = np.expand_dims(Label_one_hot_val, axis=0)

            else:
                img_dir = os.path.join(img_dir, 'normal')
                mat_list = os.listdir(img_dir)
                mat_name = random.choice(mat_list)
                img_dir = os.path.join(img_dir, mat_name)
                data = scipy.io.loadmat(img_dir)
                data = data['normal_Array']
                data = (data - data.min()) / (data.max() - data.min())

                w = data.shape[1]
                h = data.shape[0]

                expend = math.ceil(224 / w)
                data = np.tile(data, (1, expend))
                data_resize = data[:, 0:224]
                val_Img = np.resize(data_resize, (224, 224))

                val_Img = np.expand_dims(val_Img, axis=0)
                val_Img = np.expand_dims(val_Img, axis=3)

                Label_one_hot_val = np.array([1, 0])
                Label_one_hot_val = np.expand_dims(Label_one_hot_val, axis=0)



            validata_feed = {x: val_Img, label_one_hot: Label_one_hot_val}
            training_feed = {x: Img, label_one_hot: Label_one_hot_train}
            validata_acc = sess.run(accuracy,feed_dict=validata_feed)
            training_acc = sess.run(accuracy,feed_dict=training_feed)
            #####summary
            all_validata_acc = all_validata_acc + validata_acc
            all_training_acc = all_training_acc + training_acc

            if i % 500 == 0:
                Mean_acc_val = all_validata_acc / 500
                Mean_acc_train = all_training_acc / 500
                summary = sess.run(merge_writer,feed_dict={mean_acc:Mean_acc_val, x: val_Img, label_one_hot: Label_one_hot_val})
                writer_valid.add_summary(summary, global_step=sess.run(global_step))
                summary = sess.run(merge_writer,feed_dict={mean_acc:Mean_acc_train, x: Img, label_one_hot: Label_one_hot_train})
                writer.add_summary(summary, global_step=sess.run(global_step))
                # if Mean_acc_val>=0.88:
                save_path = saver.save(sess,
                                       "/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/ill/5_cross_vaildation/model/{}/model.ckpt".format(str(val_num)),
                                       global_step=sess.run(global_step))


                print("*"*20)
                print('After {0} training epochs,{1} training steps, training accuracy is {2}'.format(epoch, i, training_acc))
                print('After {0} training epochs,{1} training steps, validation accuracy is {2}'.format(epoch, i, validata_acc))
                print("*"*20)

                all_validata_acc=0
                all_training_acc=0