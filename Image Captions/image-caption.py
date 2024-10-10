from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import cv2

#Link for the model : https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

# load the model 
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Extract the features
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# check the Cuda device 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# match the device and the model 
model.to(device)

max_length = 16
num_beams = 4 
gen_kwargs = {"max_length":max_length, "num_beams": num_beams}


# Sending several images to the funcion

def predict_the_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)

        # Convert to RGB
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    # use the feature extractor and get the words as the result
    pixel_values = feature_extractor(images = images , return_tensors="pt").pixel_values

    print("**************")
    print("pixel_values :")
    print(pixel_values)

    pixel_values = pixel_values.to(device)

    # Get the words
    output_ids = model.generate(pixel_values, **gen_kwargs)
    
    print("**************")
    print("output_ids :")
    print(output_ids)


    # Prediction and decode all the id's into words
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    print("**************")
    print("Final result :")
    print(preds)

    return preds



images_paths = ["Python-Code-Cool-Stuff\Image Captions\Dori.jpg", "Python-Code-Cool-Stuff\Image Captions\haverim.jpg"]
results = predict_the_caption(images_paths)


# Display the result :

for image_path , text in zip(images_paths, results):
    image = cv2.imread(image_path)

    cv2.putText(image , text , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)


cv2.destroyAllWindows()
