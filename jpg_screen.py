import numpy as np
import tensorflow as tf
import sys
#限制TensorFlow的警告信息的输出
import os
#默认为0：输出所有log信息
#设置为1：进一步屏蔽INFO信息
#设置为2：进一步屏蔽WARNING信息
#设置为3：进一步屏蔽ERROR信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#npy载入并处理：提取出有病的数据，并且提取出和有病样本三倍多的正常样本
def npy_load(x_info,y_info):
    x = np.load(x_info)
    y = np.load(y_info)
    #阳性样本总和
    total_disease = len(np.where(y==1)[0]) + len(np.where(y==2)[0]) + len(np.where(y==3)[0])
    x_posi = x[np.where(y==0)[0][:total_disease*3]]
    x_1,x_2,x_3 = x[np.where(y==1)[0]], x[np.where(y==2)[0]], x[np.where(y==3)[0]]
    return x_posi,x_1,x_2,x_3

#图像增强处理
def transform_array(arrays):
    list_array = []
    for array in arrays:
        #图像左右颠倒
        list_array.append(np.array(tf.image.flip_left_right(array)).astype('float32'))
        #图像上下颠倒
        list_array.append(np.array(tf.image.flip_up_down(array)).astype('float32'))
        #图像对比度下降
        list_array.append(np.array(tf.image.adjust_contrast(array,0.7)).astype('float32'))
        #图像JPG质量下降
        list_array.append(np.array(tf.image.adjust_jpeg_quality(array,20)).astype('float32'))
        #图像左右上下颠倒
        list_array.append(np.array(tf.image.flip_left_right(tf.image.flip_up_down(array))).astype('float32'))
        #图像左右上下颠倒后对比度下降
        list_array.append(np.array(tf.image.adjust_contrast(tf.image.flip_left_right(tf.image.flip_up_down(array)),0.7)).astype('float32'))
        #图像左右上下颠倒后JPG质量下降
        list_array.append(np.array(tf.image.adjust_jpeg_quality(tf.image.flip_left_right(tf.image.flip_up_down(array)),20)).astype('float32'))
        #图像左右颠倒后对比度下降
        list_array.append(np.array(tf.image.adjust_contrast(tf.image.flip_left_right(array),0.7)).astype('float32'))
        #图像上下颠倒后JPG质量下降
        list_array.append(np.array(tf.image.adjust_jpeg_quality(tf.image.flip_up_down(array),20)).astype('float32'))
        #图像对比度下降后JPG质量下降
        list_array.append(np.array(tf.image.adjust_jpeg_quality(tf.image.adjust_contrast(array,0.7),20)).astype('float32'))
    return np.array(list_array)

def main():
    prefix = sys.argv[1]
    x_posi,x_1,x_2,x_3 = npy_load(sys.argv[2],sys.argv[3])
    #增多阳性样本
    x_3 = transform_array(x_3)
    x_2 = transform_array(x_2)
    y = np.array([0]*len(x_posi) + [1]*len(x_1) + [2]*len(x_2) + [3]*len(x_3))
    x = np.concatenate((x_posi,x_1,x_2,x_3),axis=0)
    #保存数组 x,y
    np.save(prefix+'_x_trans.npy',x)
    np.save(prefix+'_y_trans.npy',y)

help='''usage:
python3.6 %s <out_file_prefix> <x_npy> <y_npy>'''%sys.argv[0]

if len(sys.argv) !=4:
    print(help)
    sys.exit(0)
main()