#!/usr/bin/env python
# _*_ coding: utf-8
# author linbin date 2018-04-13
####由于一个txt文件编码出问题了，所以直接open读取的时候显示乱码。
####本程序是先用 对应的格式解码读取，再提取想要的信息，然后用utf-8的格式
####保存。
### python使用codecs模块进行文件操作-读写中英文字符
### https://blog.csdn.net/chenyxh2005/article/details/72465758#t0

import codecs
file_name = "C:\\Users\\lxy\\Desktop\\MEG-2-11.txt"
txtName = "C:\\Users\\lxy\\Desktop\\pic2.txt"
f_name = codecs.open(txtName, "w", 'utf-8')
with codecs.open(file_name, 'r', 'utf-16-le') as file_to_read:
    while True:
        lines = file_to_read.readline()  # 整行读取数据
        if not lines:
            break
        lines = lines.strip(' ')
        true_index = lines.find('simulas:')
        if true_index != -1:
            pic_name = lines[11:]
            f_name.write(pic_name)
f_name.close()
