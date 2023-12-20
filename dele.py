import os
import cv2
# ftest = open('data/test.txt')
# path = 'data/samples'

# for test in ftest:
#     img = cv2.imread(test.split( )[0])
#     newpath = os.path.join(path,test.split('\n')[0].split('/')[2].split('.')[0])
#     print(newpath+'.jpg')
#     cv2.imwrite(newpath+'.jpg',img)
# ftest.close()
img = cv2.imread('/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/attention_yolo/yolov3-defect-detection/data/images/000001.jpg')
print(img.shape)

