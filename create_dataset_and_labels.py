import os
import cv2
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/YasinDurusoy/Desktop/Projects/Teknofest/sonveriler.csv")

namelist = df['dosyaadi'].values.tolist()
classlist = df['class'].values.tolist()
xminlist = df['xmin'].values.tolist()
xmaxlist = df['xmax'].values.tolist()
yminlist = df['ymin'].values.tolist()
ymaxlist = df['ymax'].values.tolist()
xcen = []
ycen = []
weight = []
height = []
for i in range(len(xminlist)):
    xminlist[i] =xminlist[i] / 299
    xmaxlist[i] =xmaxlist[i] / 299
    yminlist[i] = yminlist[i] / 299
    ymaxlist[i] = ymaxlist[i] / 299
    xcen.append(round((xminlist[i]+xmaxlist[i])/2,5))
    ycen.append(round((yminlist[i]+ymaxlist[i])/2,5))
    weight.append(round((xmaxlist[i]-xminlist[i]),5))
    height.append(round((ymaxlist[i]-yminlist[i]),5))

homedir ="C:/Users/YasinDurusoy/Desktop/Projects/Teknofest/dosyalar"
imlist = []
for image in os.listdir(homedir):
    imlist.append(image)

print("HERE")
for i in range(len(imlist)):
    name = imlist[i].split('.')
    name = name[0] + '.txt'
    #print(name)
    for j in range(len(namelist)):
        if imlist[i] == namelist[j]:
            if classlist[j] != 0:
                f = open("C:/Users/YasinDurusoy/Desktop/abdominal-detection-pycharm/abdominal-dataset-yolo/labels/" + name,'w')
                write_line = str(classlist[j]-1)+' '+str(xcen[j])+' '+str(ycen[j])+' '+str(weight[j])+' '+str(height[j])+'\n'
                #print(write_line)
                f.write(write_line)
                impath = os.path.join("C:/Users/YasinDurusoy/Desktop/Projects/Teknofest/dosyalar",namelist[j])
                img2save = cv2.imread(impath)
                savepath = os.path.join("C:/Users/YasinDurusoy/Desktop/abdominal-detection-pycharm/abdominal-dataset-yolo/images/",namelist[j])
                cv2.imwrite(savepath, img2save)
                f.close()
                
import os
from sklearn.model_selection import train_test_split
import shutil

imdir = 'C:/Users/YasinDurusoy/Desktop/abdominal-detection-pycharm/abdominal-dataset-yolo/images/'
labeldir = 'C:/Users/YasinDurusoy/Desktop/abdominal-detection-pycharm/abdominal-dataset-yolo/labels/'
# Read images and annotations
images = [os.path.join(imdir, x) for x in os.listdir(imdir)]
annotations = [os.path.join(labeldir, x) for x in os.listdir(labeldir) if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.1, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

homedir = 'C:/Users/YasinDurusoy/Desktop/abdominal-detection-pycharm/abdominal-dataset-yolo/'
os.mkdir(homedir+'images/train')
os.mkdir(homedir+'images/val')
os.mkdir(homedir+'images/test')
os.mkdir(homedir+'labels/train')
os.mkdir(homedir+'labels/val')
os.mkdir(homedir+'labels/test')

# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Move the splits into their folders
move_files_to_folder(train_images, homedir+'images/train')
move_files_to_folder(val_images, homedir+'images/val/')
move_files_to_folder(test_images, homedir+'images/test/')
move_files_to_folder(train_annotations, homedir+'labels/train/')
move_files_to_folder(val_annotations, homedir+'labels/val/')
move_files_to_folder(test_annotations, homedir+'labels/test/')