import tensorflow as tf
import numpy as np
from model import multy_model_build, multy_resnet50_build, multy_DenseNet201_build, multy_InceptionV3_build, multy_ResNet50V2_build, multy_InceptionResNetV2_build, multy_Xception_build
from sklearn.metrics import accuracy_score
import sys,os

def dir_check(dir):
    if not os.path.exists(dir):os.mkdir(dir)

def model_build(model,checkpoint_path,train_epoch,batch_size,x_train,y_train):
    dir_check(checkpoint_path)
    checkpoint_path = '%s/TCT_pathology.{epoch:02d}-{val_loss:.2f}.ckpt'%checkpoint_path
    #monitor：需要监视的值;verbose：信息展示模式;save_best_only：当设置为True时，将只保存在验证集上性能最好的模型;mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断;save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）;period：CheckPoint之间的间隔的epoch数
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=1,save_freq='epoch',monitor='val_loss',save_best_only=True,period=20),
                #ReduceLROnPlateau当评价指标不在提升时，减少学习率，语法如下：
                #monitor：被监测的量;factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少;patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发;mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少;epsilon：阈值，用来确定是否进入检测值的“平原区”;cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作;min_lr:学习率的下限
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.00000001, cooldown=0, min_lr=0)]
    

    #自制模型
    # model = multy_model_build(checkpoint_path)
    
    train_history = model.fit(x_train,y_train,
                                validation_split=0.2,
                                epochs = train_epoch,
                                batch_size = batch_size,
                                verbose=2,
                                shuffle=True,
                                callbacks=callbacks)
    print(train_history.history)
    return model

def validation(model):
    x_val = np.load('/project/user/chenqi/TCT_AI/TCT_classify/x_trans.npy')
    y_val = np.load('/project/user/chenqi/TCT_AI/TCT_classify/y_trans.npy')
    y_predict = model.predict_classes(x_val).astype('int8')
    accuracy = accuracy_score(y_val,y_predict)
    print('accuracy:%s%%'%(accuracy*100))

def main():
    x = np.load(sys.argv[1])
    y = np.load(sys.argv[2])
    train_epoch = 200
    batch_size = 300

    input_shape = (450,300,3)

    #ResNet50模型
    # print('####################################################ResNet50 begin#########################################')
    # checkpoint_path = 'ResNet50'
    # model_ResNet50 = multy_resnet50_build(checkpoint_path,input_shape)
    # model_ResNet50 = model_build(model_ResNet50,checkpoint_path,train_epoch,batch_size,x,y)
    # validation(model_ResNet50)

    print('####################################################DenseNet201 begin#########################################')
    #DenseNet201模型
    checkpoint_path = 'DenseNet201'
    model_DenseNet201 = multy_DenseNet201_build(checkpoint_path,input_shape)
    model_DenseNet201 = model_build(model_DenseNet201,checkpoint_path,train_epoch,batch_size,x,y)
    validation(model_DenseNet201)

    print('####################################################InceptionV3 begin#########################################')
    #InceptionV3
    checkpoint_path = 'InceptionV3'
    model_InceptionV3 = multy_InceptionV3_build(checkpoint_path,input_shape)
    model_InceptionV3 = model_build(model_InceptionV3,checkpoint_path,train_epoch,batch_size,x,y)
    validation(model_InceptionV3)

    print('####################################################ResNet50V2 begin#########################################')
    #ResNet50V2
    checkpoint_path = 'ResNet50V2'
    model = multy_ResNet50V2_build(checkpoint_path,input_shape)
    model = model_build(model,checkpoint_path,train_epoch,batch_size,x,y)
    validation(model)
    
    # print('####################################################InceptionResNetV2 begin#########################################')
    # #InceptionResNetV2
    # checkpoint_path = 'InceptionResNetV2'
    # model = multy_InceptionResNetV2_build(checkpoint_path,input_shape)
    # model = model_build(model,checkpoint_path,train_epoch,batch_size,x,y)
    # validation(model)

    print('####################################################Xception begin#########################################')
    #Xception
    checkpoint_path = 'Xception'
    model = multy_Xception_build(checkpoint_path,input_shape)
    model = model_build(model,checkpoint_path,train_epoch,batch_size,x,y)
    validation(model)

main()
