import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model , preprocess = clip.load("ViT-B/32", device = device)

textLabels = ["Kids eating ice cream" , "A cute dog", "Crowd in a concert"]

#load the test image

testImagePath = "image1.jpg"
image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

# convert the labels and prepare it for the model
text = clip.tokenize(textLabels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image , logits_per_text = model(image,text)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print(probs)

#display the text lables

probs = probs[0]
answer = np.argmax(probs)
text = textLabels[answer]

print ('Predicted : ' + text)

#show the image with the predicted text using OpenCV

img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img , text , (0,100), font , 2 , (209,19,77), 3)

cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()




