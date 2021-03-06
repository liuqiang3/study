from img_rf_fusion import ResNet50_fusion_v1
from keras.layers import GlobalAveragePooling2D ,Dense
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf
import keras
import math
import random
import numpy as np
import glob
import scipy.io
import cv2
from imgaug import augmenters as iaa
import os
import keras.layers as layers
import keras.models as models
import keras.backend as backend


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


def ResNet50(input_shape=(224,224,3),classes=2):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3


    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
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

    # x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predict = layers.Dense(2, activation='softmax')(x)
    inputs = img_input
    # Create model.
    model = models.Model(inputs, predict, name='resnet50')


    return model

def ResNet50_v2(input_shape=(224,224,3),classes=2):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3


    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
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

    # x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predict = layers.Dense(2, activation='softmax')(x)
    inputs = img_input
    # Create model.
    model = models.Model(inputs, predict, name='resnet50')


    return model

Img_3c =np.zeros([224,224,3])
Img_3c_val =np.zeros([224,224,3])

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


def data_precess(img_path_ill,imgList_ill,array='ill_Array',mode='train'):

    img_name = random.choice(imgList_ill)
    img_dir = os.path.join(img_path_ill, img_name)
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
    img_dir_img = os.path.join(img_dir, mat_name_img)
    data_img = cv2.imread(img_dir_img)
    data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min())
    if mode == 'train':
        # Img_img = augmentNumpy(data_img)
        Img_img = data_img

    else :
        Img_img = data_img

    return Img_mat,Img_img
def batch_img(img_path_ill,imgList_ill,img_path_normal,imgList_normal,batch_size=8,mode = 'train'):
    batch_images = []
    batch_labels = []
    batch_images_2 = []
    Img_3c = np.zeros([batch_size, 224, 224, 3])
    for i in range(batch_size):
        seed_random = random.randint(0,1)
        if seed_random == 0:
            Img_mat,Img_img = data_precess(img_path_ill, imgList_ill, array='ill_Array',mode =mode)

            Label_one_hot_train = np.array([0,1])
        else:
            Img_mat, Img_img = data_precess(img_path_normal,imgList_normal, array='normal_Array',mode=mode)

            Label_one_hot_train = np.array([1,0])

        batch_images.append(Img_mat)
        batch_images_2.append(Img_img)

        batch_labels.append(Label_one_hot_train)
    batch_images = np.array(batch_images)
    batch_images_2 = np.array(batch_images_2)

    batch_labels = np.array(batch_labels)
    Img_3c[:, :, :, 0] = batch_images
    Img_3c[:, :, :, 1] = batch_images
    Img_3c[:, :, :, 2] = batch_images
    Img_3c_2 = batch_images_2

    return Img_3c,batch_labels,Img_3c_2

model = ResNet50(input_shape=(224,224,3),classes=2)

model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy')
epochs = 200
batchs = 1000
img_path_ill = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF4/data_add_normal/ill'
img_path_normal = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF4/data_add_normal/normal'

print(model.summary())

imgList_normal = read_data(train_set=[ 2, 3, 4,5],
                           data_path=img_path_normal)
imgList_ill = read_data(train_set=[ 2, 3, 4,5],
                        data_path=img_path_ill)

val_imgList_normal = read_data(train_set=[1],
                           data_path=img_path_normal)
val_imgList_ill = read_data(train_set=[1],
                        data_path=img_path_ill)



logdir_train = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/resnet/record/train'
logdir_test = '/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/resnet/record/test'

# writer_train = tf.summary.FileWriter(logdir_train)
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.2,height_shift_range=0.2)
# train_generator = train_datagen.flow_from_directory(directory='',target_size=(224,224),batch_size=64)
# with tf.Session() as sess:
def acc(predict,label):
    corret_pred = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
    accuracy = np.mean(corret_pred)
    return accuracy
all_acc = 0
all_acc_val = 0
writer = tf.summary.FileWriter(logdir_train)
writer_val = tf.summary.FileWriter(logdir_test)
n = 0
for epoch in range(epochs):
    for i in range(batchs):
        n += 1
        seed_random = random.randint(0,1)
######################## dataread
        Img_3c_final, Label_one_hot_train,Img_3c_final_2 = batch_img(img_path_ill=img_path_ill,
                                            imgList_ill=imgList_ill,
                                            img_path_normal=img_path_normal,
                                            imgList_normal=imgList_normal,
                                            batch_size=16,
                                            )

        Img_3c_final_val, Label_one_hot_train_val,Img_3c_final_val_2 = batch_img(img_path_ill=img_path_ill,
                                            imgList_ill=val_imgList_ill,
                                            img_path_normal=img_path_normal,
                                            imgList_normal=val_imgList_normal,
                                            batch_size=16,
                                            mode='test')
        cv2.imwrite('/mnt/data/VGG_ultrasound/11111/{}.jpg'.format(i),Img_3c_final[0]*255)