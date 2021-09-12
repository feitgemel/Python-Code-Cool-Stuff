# please follow first the install settings before coding , like creating the conda enviroment
# important - the Python vscode home directory shold be the Deoldify directory 

from deoldify import device 
from deoldify.device_id import DeviceId

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

# this is the main object for colorizing image
colorizer = get_image_colorizer(artistic=True)

# this parameter is very memory consume, So if you have memory issues running this Python script
# try to reduce it to the minimum and raise as trail and error process
# to reduce the running time , I changed it for this demo to the minimum : 7 

render_factor =  7 # 35 # this input parameter is minimum 7 and maximum 40
watermakred = False # this paramter coltrols watermark in the result image


# lets load an image from a URL (not localy)

#This is the link for the Black and white image 
source_url = 'https://images.pexels.com/photos/53453/marilyn-monroe-woman-actress-pretty-53453.jpeg'


image_path = colorizer.plot_transformed_image_from_url(url=source_url,render_factor= render_factor, compare=True , watermarked= watermakred )
#image_path = colorizer.plot_transformed_image(path='C:/GitHub/Python-Code-Cool-Stuff/DeOldify/media/testBW.jpg',render_factor=render_factor, compare=True, watermarked=watermakred)

# all the rest should be the same 
# lets run it

# the image is too big 
# resolution is : 	2345px x 3313px
# lets resuce the size of the image with 60%

print('this is the result image path :')
print(image_path)

#lets show the image
image = cv2.imread(str(image_path))

scale = 60
width = int(image.shape[1] * scale / 100)
height = int(image.shape[0] * scale / 100)
dim = (width, height)

resized = cv2.resize(image , dim , interpolation = cv2.cv2.INTER_AREA)


cv2.imshow('image',resized)
cv2.waitKey(0)

