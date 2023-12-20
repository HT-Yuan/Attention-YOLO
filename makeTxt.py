import os
import random

testval_percent = 0.3
# 相对testval集合中test所占的比值
test_percent = 0.5

xmlfilepath = 'data/Annotations'
txtsavepath = 'data/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
print (num)
list = range(num)
# 1350 * 0.3 = 313
testval_num = int(num * testval_percent)
print (testval_num)
# 202
test_num = int(testval_num * test_percent)
print (test_num)
# 202个从num合集中选择的随机数
testval = random.sample(list, testval_num)

test = random.sample(testval, test_num)

ftest = open('data/ImageSets/test.txt', 'w')
ftrain = open('data/ImageSets/train.txt', 'w')
fval = open('data/ImageSets/val.txt', 'w')
a = 0
for i in list:
    name = total_xml[i][:-4] + '\n' # delete last 4 char
    if i in testval:
        a +=1
        if i in test:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
print(a)
ftrain.close()
fval.close()
ftest.close()