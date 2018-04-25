#!/usr/bin/env python
# _*_ coding: utf-8
# author linbin Date 2018-04-25

### 控制脚本处理数据  每次处理一个被试
### 调用shell命令，比如本文代码是到处CPU的使用率 和 删除生成的文件

import os
os.system("top -bn 1 -i -c > statusll.log")
os.remove("statusll.log")