import os
import os.path
from os import path

def cp(f,dir):
  import shutil
  shutil.copy2(f,dir)

directory = r'/home/user/src/darknet/proj/eb/saleboxes'

for i in range(3):
  names = []
  for filename in os.listdir(directory):
    names.append('proj/eb/saleboxes/' + filename)
  if path.exists(r'/home/user/src/darknet/pathnames2.txt'):
    f = open('pathnames2.txt', 'r+')
    f.truncate(0) # need '0' when using r+
    f.write('\n'.join(names))
  else:
    f = open("pathnames2.txt", "w")
    f.write('\n'.join(names))

#copy correct cfg file to cfg folder
cp(r'proj/eb/cfg/tiny2/v2/yolov4-tiny-obj.cfg', r'/home/user/src/darknet/cfg')
cp(r'proj/eb/cfg/tiny2/v2/obj.names', r'/home/user/src/darknet/data')
cp(r'proj/eb/cfg/tiny2/v2/obj.data', r'/home/user/src/darknet/data')

#run first detections
os.system('./darknet detector test data/obj.data cfg/yolov4-tiny-obj.cfg proj/eb/cfg/tiny2/v2/yolov4-tiny-obj_final.weights -thresh 0.9 -ext_output -dont_show -out </home/user/src/darknet/pathnames2.txt result.json')
