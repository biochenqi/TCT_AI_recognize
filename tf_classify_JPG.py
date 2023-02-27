# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import PIL.Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os,argparse,sys
from matplotlib.font_manager import FontProperties
from sklearn.metrics import accuracy_score,recall_score,precision_score
from model import model_build

#使用中文字体
zhfont = FontProperties(fname = '/usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf')

usages = """ 
Author : chenqi
Email  : 
Date   : 
Version: v1.0
Description:
    用来对TCT病理学图片进行机器学习分类
Example:
    python3 %s -o <outdir> --train_file <train file>|--numpy_array [x_total,y_total] --checkpoint_path <dir> --train_epoch <int> --batch_size <int>
"""%(sys.argv[0])

class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

#参数设置
def getopt():
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, description=usages)
    parser.add_argument('-o','--outdir',help='output dir,default:current dir',default='.',dest='outdir',type=str)
    parser.add_argument('--checkpoint_path',help='the path of checkpoint,default:checkpoint',default='checkpoint',dest='checkpoint_path',type=str)
    parser.add_argument('--train_epoch',help='the epoch of trainning,default:50',default=50,type=int,dest='train_epoch')
    parser.add_argument('--batch_size',help='the size of each batch,default:100',default=100,type=int,dest='batch_size')
    parser.add_argument('--train_file',help='train_file',dest='train_file',type=str)
    parser.add_argument('--numpy_array',help='images and labels',dest='numpy_array',type=str,nargs=2)
    args = parser.parse_args()
    if not args.train_file and not args.numpy_array:
        print('train file or numpy array must be given!')
        sys.exit(0)
    return args

#载入数据及处理
def jpg_input_deal(list_jpg):
    y = int(list_jpg[0])
    x_1 = pixel_reduction(list_jpg[1],300)
    x_2 = pixel_reduction(list_jpg[2],300)
    x = np.concatenate((x_1,x_2),axis=0)
    return x,y


#图像数字标准化以及处理
def pixel_reduction(file_name,max_dim=None):
    img = PIL.Image.open(file_name)
    if max_dim:
        img.thumbnail((max_dim,max_dim))
    
    return np.array(img)

#载入权重
def load_weight(checkpoint_path):
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    return latest

#构建模型
def model_build_old(weight_save_file):
    latest = load_weight(weight_save_file)
    model = tf.keras.models.Sequential()
    #第一层卷积层
    model.add(tf.keras.layers.Conv2D(filters=32,
                                 kernel_size=(3,3),
                                 input_shape=(450,300,3),
                                 activation='relu',
                                 padding='same'))
    #防止过拟合 dropout随机丢弃30%神经元
    model.add(tf.keras.layers.Dropout(rate=0.3))
    #第一个池化层
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3,3),
                                 activation='relu',
                                 padding='same'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    #平坦层
    model.add(tf.keras.layers.Flatten())
    #添加输出层
    model.add(tf.keras.layers.Dense(4,activation='softmax'))

    #设置训练模式
    model.compile(optimizer= 'Adam',
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])
    if latest:
        model.load_weights(latest)

    return model 

#预测结果图像输出
def plot_images_labels_prediction(images,  #图像列表
                                  labels,  #标签列表
                                  preds,   #预测值列表
                                  outdir,
                                  index=0, #从第index位置开始显示
                                  num=10): #缺省一次显示1幅

    dict_result_class = {0: ['未见上皮内病变或恶性病变（NILM)。',
  '上皮细胞少，请结合临床。如炎症明显，建议治疗后复查；如怀疑宫颈上皮病变应进行阴道镜检查',
  '未见上皮内病变或恶性病变（NILM），中度炎症反应。',
  '1.未见上皮内病变或恶性病变（NILM)。2.见线索细胞大于20%，提示细菌性阴道病的可能，建议阴道PH',
  '1.未见上皮内病变或恶性病变（NILM)。2.可见真菌感染，建议治疗后复查。',
  '未见上皮内病变或恶性病变（NILM），重度炎症反应。',
  '1.未见上皮内病变或恶性病变（NILM)。2.上皮细胞少，建议必要时复查。',
  '1.未见上皮内病变或恶性病变（NILM)。2.细胞萎缩，上皮细胞少，建议定期复查。',
  '1.非典型鳞状上皮细胞，意义不明确（ASC－US），细胞形态改变提示HPV感染可能。2.建议行HPV检测',
  '1.未见上皮内病变或恶性病变（NILM)。2.可见滴虫感染，建议治疗后复查。',
  '未见上皮内病变或恶性病变（NILM），轻度炎症反应。'],
 1: ['1.非典型鳞状上皮细胞，意义不明确（ASC-US）。2.建议2-3个月后复查 。',
  '1.非典型鳞状上皮细胞，不排除高级别鳞状上皮内病变（ASC-H）。2.建议宫颈活检。',
  '上皮细胞少，建议重取复查。',
  '1.非典型腺细胞（AGC-NOS）。2.建议宫颈活检+宫颈管分段诊刮。'],
 2: ['1.低级别鳞状上皮内病变（LSIL）。2.建议宫颈活检。',
  '1.低级别鳞状上皮内病变（LSIL），细胞形态改变提示HPV感染可能。2.建议行宫颈活检及HPV检测。'],
 3: ['1.高级别鳞状上皮内病变（HSIL）。2.建议宫颈活检。']}
    fig = plt.gcf() #获取当前图片  get current figure
    fig.set_size_inches(10,4)
    if num >10:
        num = 10 #最多显示10个子图
    for i in range(0,num):
        ax = plt.subplot(2,5,i+1)
        ax.imshow(images[index])
        # title = dict_result_class[labels[index]][0] + ":" + str(labels[index])
        title = str(labels[index])
        if len(preds) > 0 :
            title += '==>' + str(preds[index])
        ax.set_title(title,fontsize=10,fontproperties=zhfont)
        #不显示横纵坐标
        ax.set_xticks([])
        ax.set_yticks([])
        
        index += 1

    plt.savefig('%s/result.png'%outdir)

def array_data_deal(array_x,array_y):
    index_1 = np.where(array_y==1)[0]
    array_y = np.delete(array_y,index_1)
    #删除多维数组的index行
    array_x = np.delete(array_x,index_1,axis=0)
    array_y[np.where(array_y==2)[0]] = 1
    array_y[np.where(array_y==3)[0]] = 1
    return array_x,array_y

if __name__ == '__main__':
    args = getopt()
    checkpoint_path = '%s/TCT_pathology.{epoch:02d}-{val_loss:.2f}.ckpt'%args.checkpoint_path
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_freq='epoch'),
    #                 #防止过拟合，当损失开始连续变大3次就停止
    #                 tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)]
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_freq='epoch',monitor='val_loss')]

    if args.train_file:
        with open(args.train_file,'r') as f:
            x_total,y_total = [],[]
            for line in f:
                line = line.strip().split('\t')
                image,label = jpg_input_deal(line)
                x_total.append(image)
                y_total.append(label)
        x_total = np.array(x_total)
        y_total = np.array(y_total)
        x_total = x_total.astype('float32')/255.0
        #保存数组 x,y
        np.save('x_info.npy',x_total)
        np.save('y_info.npy',y_total)
    elif args.numpy_array:
        x_total = np.load(args.numpy_array[0])
        y_total = np.load(args.numpy_array[1])
        if x_total.dtype != np.float32:
            x_total = np.array(x_total,dtype=np.float32)
    y_total = y_total.astype('int8')
    x_total,y_total = array_data_deal(x_total,y_total)

    ###test delte 1
    # index_1 = np.where(y_total==1)[0]
    # y_total = np.delete(y_total,index_1)
    # #删除多维数组的index行
    # x_total = np.delete(x_total,index_1,axis=0)

    x_train,x_test,y_train,y_test = train_test_split(x_total,y_total,test_size=0.2)
    x_total,y_total = 0,0
    model = model_build(checkpoint_path)
    train_history = model.fit(x_train,y_train,
                                validation_split=0.2,
                                epochs = args.train_epoch,
                                batch_size = args.batch_size,
                                verbose=2,
                                callbacks=callbacks)
    print(train_history.history)
    y_predict = model.predict_classes(x_test).astype('int8')
    # plot_images_labels_prediction(x_test,y_test,y_predict,args.outdir)
    accuracy = accuracy_score(y_test,y_predict)
    # sensitive = recall_score(y_test,y_predict)
    # precision = precision_score(y_test,y_predict)
#     print('''accuracy:%.4f%%
# sensitive:%.4f%%
# precision:%.4f%%'''%(accuracy*100,sensitive*100,precision*100))
    print('accuracy:%s%%'%(accuracy*100))