import tensorflow as tf
import cv2
import numpy as np
import os
import random
from keras.optimizers import SGD
import scipy.io
import matplotlib.pyplot as plt
import math
import keras
import keras.layers as layers
import keras.models as models
import keras.backend as backend
from img_rf_fusion import ResNet50_fusion_v2

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
def ResNet50_fusion_v1(input_shape=(224,224,3),classes=2):
    img_input_rf = layers.Input(shape=input_shape)

    bn_axis = 3
################ branch1:rf
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input_rf)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)

########### branch2:img
    img_input_img = layers.Input(shape=input_shape)
    bn_axis = 3
    y = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad_img')(img_input_img)
    y = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1_img')(y)
    y = layers.BatchNormalization(axis=bn_axis, name='bn_conv1_img')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D((3, 3), strides=(2, 2))(y)

    y = conv_block(y, 3, [64, 64, 256], stage=2, block='a_img', strides=(1, 1))
    y = identity_block(y, 3, [64, 64, 256], stage=2, block='b_img')
    y = identity_block(y, 3, [64, 64, 256], stage=2, block='c_img')

    y = conv_block(y, 3, [128, 128, 512], stage=3, block='a_img')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='b_img')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='c_img')
    y = identity_block(y, 3, [128, 128, 512], stage=3, block='d_img')

    y = conv_block(y, 3, [256, 256, 1024], stage=4, block='a_img')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='b_img')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='c_img')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='d_img')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='e_img')
    y = identity_block(y, 3, [256, 256, 1024], stage=4, block='f_img')

    y = conv_block(y, 3, [512, 512, 2048], stage=5, block='a_img')
    y = identity_block(y, 3, [512, 512, 2048], stage=5, block='b_img')
    y = identity_block(y, 3, [512, 512, 2048], stage=5, block='c_img')

    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(512, activation='relu')(y)
########## fusion
    z = layers.concatenate([x, y], axis=1)
    predict = layers.Dense(2, activation='softmax')(z)
    model = models.Model([img_input_rf,img_input_img], predict, name='resnet50')
    return model
def acc(predict,label):
    corret_pred = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
    accuracy = np.mean(corret_pred)
    return accuracy
keras.backend.set_learning_phase(0)
model = ResNet50_fusion_v2(input_shape=(224,224,3),classes=2)

model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy')
all_acc = 0
all_acc_val = 0

def data_precess_test(img_dir,array='ill_Array',mode='train'):

    # img_name = random.choice(imgList_ill)
    # img_dir = os.path.join(img_path_ill, img_name)
    data_list = os.listdir(img_dir)
    filterfuc_mat = lambda x: x.split('.')[-1] == 'mat'
    mat_list = [element for element in data_list if filterfuc_mat(element)]

    mat_name_mat = random.choice(mat_list)
    img_dir_mat = os.path.join(img_dir, mat_name_mat)
    data_mat = scipy.io.loadmat(img_dir_mat)
    data_mat = data_mat[array]

    data_mat = (data_mat - data_mat.min()) / (data_mat.max() - data_mat.min())
    w = data_mat.shape[1]
    h = data_mat.shape[0]
    # expend = math.ceil(224 / w)
    # data_mat = np.tile(data_mat, (1, expend))
    # data_mat_resize = data_mat[:, 0:224]
    # Img_mat = cv2.resize(data_mat_resize, (224, 224))

    expend = math.ceil(224 / w)

    blank = abs(math.ceil((224 - h) / 2))

    if h <= 224:

        blank_array = np.zeros((blank, w))


        data_mat_concatenate = np.concatenate((blank_array, data_mat, blank_array), axis=0)
    else:
        data_mat_concatenate = data_mat[blank:blank+224,:]
    data_mat = np.tile(data_mat_concatenate, (1, expend))
    Img_mat = data_mat[0:224, 0:224]

    ## img_roi

    filterfuc_img = lambda x: x.split('.')[-1] == 'jpg'
    img_list = [element for element in data_list if filterfuc_img(element)]
    img_list = [element for element in img_list if element.split('_')[1]==array.split('_')[0]]
    mat_name_img = random.choice(img_list)
    print(mat_name_img)

    img_dir_img = os.path.join(img_dir, mat_name_img)
    data_img = cv2.imread(img_dir_img)
    data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min())
    if mode == 'train':
        # Img_img = augmentNumpy(data_img)
        Img_img = data_img

    else :
        Img_img = data_img

    return Img_mat,Img_img
ill_root_path = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF4/data_add_normal/ill/1'
normal_root_path = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF4/data_add_normal/normal/1'
Logits_list = []
Label_list = []
Img_3c_final_batch1 = np.zeros([1, 224, 224, 3])
Img_3c_final_batch2 = np.zeros([1, 224, 224, 3])
ill_pationt = os.listdir(ill_root_path)
#
all_TP = 0
all_FP = 0
all_TN = 0
all_FN = 0
i = 0
# keras.backend.set_learning_phase(1)

model.load_weights('/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/resnet/model/incept_195501.h5')
for layer in model.layers:
    print(layer.name)
    layer.trainable = False
    print(layer.trainable)
    print('**' * 10)
for i in range(20):
    for root_path,array in [(ill_root_path,'ill_Array'),(normal_root_path,'normal_Array')]:
        for filename in os.listdir(root_path):

            img_dir = os.path.join(root_path,filename)
            print(img_dir)

            if array == 'ill_Array':
                Label_one_hot_train = np.array([0, 1])
            if array == 'normal_Array':
                Label_one_hot_train = np.array([1, 0])
            Img_3c_final, Img_3c_final_2, = data_precess_test(img_dir, array=array)
            Img_3c_final_batch1[:, :, :, 0] = Img_3c_final
            Img_3c_final_batch1[:, :, :, 1] = Img_3c_final
            Img_3c_final_batch1[:, :, :, 2] = Img_3c_final

            Img_3c_final_batch2[:, :, :, :] = Img_3c_final_2
            Label_one_hot_train = np.expand_dims(Label_one_hot_train,axis=0)
            predict = model.predict_on_batch([Img_3c_final_batch1, Img_3c_final_batch2])

            logits_reshape = np.reshape(predict, [1, -1])[0][0]
            Logits_list.append(logits_reshape)
            print(predict)
            print(Label_one_hot_train)

            label_reshape = np.reshape(Label_one_hot_train, [1, -1])[0][0]
            Label_list.append(label_reshape)

            Predict = np.argmax(predict, axis=1)
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
            print(i)

precision = all_TP / (all_TP + all_FP)
recall = all_TP / (all_TP + all_FN)
sensitivity = all_TP / (all_TP + all_FN)
specificity = all_TN / (all_TN + all_FP)
ACC = (all_TP + all_TN) / (all_TP + all_TN + all_FP + all_FN)
#
print('all_TP:{}'.format(all_TP))
print('all_FP:{}'.format(all_FP))
print('all_TN:{}'.format(all_TN))
print('all_FN:{}'.format(all_FN))
#
print('ACC:{}'.format(ACC))

print('the precision is [{}]'.format(precision))
print('the recall is [{}]'.format(recall))
print('the sensitivity is [{}]'.format(sensitivity))
print('the specificity is [{}]'.format(specificity))
auc_data = zip(Logits_list,Label_list)
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
########### AUC
ranks = 0
M = 0
N = 0
for i,(score,label) in enumerate(sorted(auc_data,key=lambda x:x[0])):
    rank = i+1
    if label ==1:
        M += 1
        ranks += rank
    else:
        N += 1
print(ranks)
AUC = (ranks-M*(M+1)/2)/(M*N)
print(AUC)