# TCT_AI_recognize
该代码用于训练TCT图像
jpg_screen.py：
  
  该代码主要用于解决数据不平衡问题，它通过左右上下、对比度下降上升等等方式增多阳性样本
  out_file_prefix：输出文件前缀
  x_npy：图像样本转化后成数值后的文件
  y_npy：图像各个样本标签文件
  
 
pdf_trans_npy.py：
  
  该代码主要用于将图像样本转换成RGB数值
  out_file_prefix：输出文件前缀
  pdf_dir：存放TCT报告pdf的目录
  jpg_dir：存放TCT对应图像文件jpg的目录
  
model.py：
  
  该代码储存了包括自制模型与公共模型ResNet50、DenseNet201、InceptionV3等在内的众多模型用于后续训练
  
train_model.py

  该代码用于训练模型
  
