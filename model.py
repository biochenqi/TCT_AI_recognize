import tensorflow as tf
import os
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception

#载入权重
def load_weight(checkpoint_path):
    latest = tf.train.latest_checkpoint(checkpoint_path)
    return latest

def multy_model_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(11,11),
                                 input_shape=input_shape,
                                 strides=4,
                                 activation='relu',
                                 padding='same'))
    # model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),strides=1))
    # model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1))
    # model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1))
    # model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=1))
    # model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4,activation='softmax'))
    model.compile(optimizer= 'Adam',
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])
    if latest:
        model.load_weights(latest)
    return model

#input_shape:(450,300,3)
def multy_resnet50_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(ResNet50(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    
    # model = ResNet50(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),weights=None,classes=4)
    # model.compile(optimizer= 'Adam',
    #           loss = tf.keras.losses.sparse_categorical_crossentropy,
    #           metrics = ['accuracy'])
    if latest:
        model.load_weights(latest)
    return model

def multy_DenseNet201_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(DenseNet201(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    if latest:
        model.load_weights(latest)
    return model

def multy_InceptionV3_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(InceptionV3(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    if latest:
        model.load_weights(latest)
    return model

def multy_ResNet50V2_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(ResNet50V2(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    if latest:
        model.load_weights(latest)
    return model

def multy_InceptionResNetV2_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(InceptionResNetV2(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    if latest:
        model.load_weights(latest)
    return model

def multy_Xception_build(weight_save_file,input_shape):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #更改ResNet50模型的输入形状以及输出类型;resnet50最后一层有1000类，所以在本例中把最后一层去掉（include_top = False）;resnet的倒数第二层是一个三维矩阵，所以无法与全连接层连接，故要pooling;weights = 'None'：从头开始训练
    model.add(Xception(input_tensor=tf.keras.layers.Input(batch_shape=(None,)+input_shape),include_top = False,pooling = 'avg'))
    model.add(tf.keras.layers.Dense(4,activation = 'softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,optimizer="Adam",metrics = ["accuracy"])
    if latest:
        model.load_weights(latest)
    return model