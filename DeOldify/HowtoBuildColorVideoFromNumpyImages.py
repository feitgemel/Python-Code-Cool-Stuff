# out numpy array of images is in the charile temp directory
# 
import numpy
from numpy import load
import cv2
colorImagesNumpyArray = load('C:/temp/charilie/colorImages.npy') 
print(colorImagesNumpyArray.shape)

# we can see the Deoldify conda environment is not compitable
# I will a diffrent enviroment that has numpy and open cv. You can do the same

#(2499, 226, 400, 3)

# so we have 2499 images . each image is has a resoltion of 226X400 with 3 color channles

# extract the dimentions of the first images (all of them are the same)
height , width , channels = colorImagesNumpyArray[0].shape
size = (width, height)
newVideoOut = cv2.VideoWriter('C:/temp/charilie/myVideo.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)

for image in colorImagesNumpyArray:
   newVideoOut.write(image)

newVideoOut.release()
cv2.destroyAllWindows() 

