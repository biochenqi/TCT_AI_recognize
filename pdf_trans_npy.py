# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:40:15 2020

@author: Administrator
"""

import os,sys,glob,fitz,PIL.Image
import numpy as np

def info_get(pdf_handel):
    check_result = 0
    for i in pdf_handel[0].getText().split('\n'):
        if i.startswith('条 形 码：'):
            bar_code = i.split('条 形 码：')[1].strip()
        
        if check_result:
            result = i.strip()
            break
            
        if i.startswith('判读意见/结果：'):check_result = 1
    
    pdf_handel.close()
    return bar_code,result

#获取pdf的bar_code信息，方便之后查找JPG文件以及获取对应样本的癌症信息 0：表示NILM未见上皮内病变或恶性病变 1：表示非典型鳞状上皮细胞，意义不明确（ASC-US） 2：低级别鳞状上皮内病变（LSIL） 3：高级别鳞状上皮内病变（HSIL）
def pdf_info_get(file_out,pdf_dir):
    w = open(file_out,'w',encoding='utf-8')
    # dict_result_class = {}
    bar_code_list = []
    count = [0,0,0,0]
    for pdf in glob.glob('%s/*.pdf'%pdf_dir):
        pdf_handel = fitz.open(pdf)
        bar_code, result = info_get(pdf_handel)
        if 'NILM' in result:
            code = 0
            count[0] += 1
        elif 'ASC-US' in result or 'ASC-H' in result or '上皮细胞少，建议重取复查。' in result or 'AGC-NOS' in result:
            code = 1
            count[1] += 1
        elif 'LSIL' in result:
            code = 2
            count[2] += 1
        elif 'HSIL' in result:
            code = 3
            count[3] += 1
        # if code not in dict_result_class:dict_result_class[code] = {}
        # if result not in dict_result_class[code]:dict_result_class[code][result] = 0
        w.write(bar_code + '\t' + str(code) + '\n')
        bar_code_list.append(bar_code + '\t' + str(code))
    w.close()
    print(count)
    return bar_code_list

#根据barcode去找到对应的jpg文件，一般一个barcode会对应两张jpg图片
def barcode_find_jpg(bar_code_info,jpg_dir,jpg_file):
    dict_total_info = {}
    # with open(sys.argv[1],'r',encoding='utf-8') as f:
    for line in bar_code_info:
        line = line.strip().split('\t')
        try:
            dict_total_info[line[0]] = [line[1]]
        except:
            print(line)

    for jepg in glob.glob(jpg_dir + '/*/*/*.JPG'):
        bar_code = jepg.strip().split('/')[-1].split('_')[0]
        if bar_code in dict_total_info:dict_total_info[bar_code].append(os.path.abspath(jepg))
    list_jpg_file = []
    with open(jpg_file,'w',encoding='utf-8') as w:
        for value in dict_total_info.values():
            if len(value) !=3:
                print(value)
                continue
            w.write('\t'.join(value)+'\n')
            list_jpg_file.append('\t'.join(value))
    return list_jpg_file

##jpg文件转化成npy文件过程
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

def jpg_to_npy(jpg_info,x_file,y_file):
    # with open(args.train_file,'r') as f:
    x_total,y_total = [],[]
    for line in jpg_info:
        line = line.strip().split('\t')
        image,label = jpg_input_deal(line)
        x_total.append(image)
        y_total.append(label)
    x_total = np.array(x_total)
    y_total = np.array(y_total)
    x_total = x_total.astype('float32')/255.0
    #保存数组 x,y
    np.save(x_file,x_total)
    np.save(y_file,y_total)

help='''usage:
python3.6 %s <out_file_prefix> <pdf_dir> <jpg_dir>'''%sys.argv[0]

def main():
    prefix = sys.argv[1]
    bar_code_list = pdf_info_get(prefix + '_barcod.txt',sys.argv[2])
    list_jpg_file = barcode_find_jpg(bar_code_list,sys.argv[3],prefix +'_jpg.txt')
    jpg_to_npy(list_jpg_file,prefix +'_x.npy',prefix +'_y.npy')

if len(sys.argv) !=4:
    print(help)
    sys.exit(0)
main()

