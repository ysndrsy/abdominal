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