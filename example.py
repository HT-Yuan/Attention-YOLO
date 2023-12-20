import glob
import xml.etree.ElementTree as ET
import tqdm
import numpy as np

from kemans import kmeans, avg_iou

ANNOTATIONS_PATH = "data/Annotations/"
#以正斜杠/这种形式可以防止反斜杠带来的转义错误
CLUSTERS = 6

def load_dataset(path):
   dataset = []
   for xml_file in tqdm.tqdm(glob.glob("{}/*xml".format(path))):
      print(xml_file)
      tree = ET.parse(xml_file)

      height = int(tree.findtext("./size/height"))
      width = int(tree.findtext("./size/width"))

      for obj in tree.iter("object"):
         xmin = int(float(obj.findtext("bndbox/xmin"))) / width
         ymin = int(float(obj.findtext("bndbox/ymin"))) / height
         xmax = int(float(obj.findtext("bndbox/xmax"))) / width
         ymax = int(float(obj.findtext("bndbox/ymax"))) / height

         dataset.append([xmax - xmin, ymax - ymin])

   return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
# print(ANNOTATIONS_PATH)
out = np.array([[10/416,13/416],  [16/416,30/416],  [33/416,23/416],  
[30/416,61/416],  [62/416,45/416],  [59/416,119/416]])
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))