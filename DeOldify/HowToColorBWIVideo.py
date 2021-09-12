# please follow first the install settings before coding , like creating the conda enviroment
# important - the Python vscode home directory shold be the Deoldify directory 


# video file is made of frames
# so will open the video frame , read each frame, colorize it , and save all the frames in one directory
# than we will open the directory and read the frames
# all the frames will be store as on numpy array , so it will be very easy to load it and convert it to a new video

# first , lets see our video 

from deoldify import device 
from deoldify.device_id import DeviceId
import time # we need to use the sleep function between each colorzie to give time for the close the memory. 

# checking for Cuda (GPU)

device.set(device=DeviceId.GPU0)

import torch
if not torch.cuda.is_available():
    print('GPU is not avaiable')

import fastai
from deoldify.visualize import *
import warnings
warnings.filterwarnings("ignore",category=UserWarning,message=".*?Your .*? set is empty.*?")

import cv2
import numpy 
from numpy import save

index = 0 # -> the image numrator
imageArray = []

# this is the main object for colorizing image
colorizer = get_image_colorizer(artistic=True)

# this parameter is very memory consume, So if you have memory issues running this Python script
# try to reduce it to the minimum and raise as trail and error process
# to reduce the running time , I changed it for this demo to the minimum : 7 


# we will use the value 7 , but I recommend you to use 35 (more memory consumption , but better quality)
render_factor =  7 # 35 # this input parameter is minimum 7 and maximum 40
watermakred = False # this paramter coltrols watermark in the result image

cap = cv2.VideoCapture('C:/GitHub/Python-Code-Cool-Stuff/DeOldify/media/charilieBW.mp4')

if (cap.isOpened() ==False):
    print('Error opening the video file')

while(cap.isOpened()):

    # lets capture each frame inside the loop
    ret , frame = cap.read()
    if ret == True:
        index = index + 1
        imageFileName = 'c:/temp/charilie/charilie'+str(index)+'.jpg'
        # save BW image
        cv2.imwrite(imageFileName,frame)
        print('extract BW frame no. ', index, ' -> save to : ', imageFileName)

        if cv2.waitKey(25) & 0xFF == ord('q') or index >=2500 :  # the index >2500 frames is just for the demo . not to wait for the all video convert
            break

        image_path = colorizer.plot_transformed_image(path=imageFileName,render_factor=render_factor, compare=True, watermarked=watermakred)

        # if you have a low memory comupter , you should add a sleep funtion in order to give time , after each colorize, to reduce the memory 
        time.sleep(5)

        tempcolorImage = cv2.imread(str(image_path))

        # build an array of color images
        imageArray.append(tempcolorImage)

cap.release()
cv2.destroyAllWindows()

colorImagesArrayNP = np.array(imageArray)

print ('Finish creating the image array , Number of images : ', colorImagesArrayNP.shape[0])

# save the numpy array
save('c:/temp/charilie/colorImages.npy',colorImagesArrayNP)
