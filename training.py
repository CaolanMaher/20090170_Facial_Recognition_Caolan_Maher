import os
import cv2
from PIL import Image
import numpy as np
import pickle

# get the path of directory of this file
DIR = os.path.dirname(os.path.abspath(__file__))

# get our training images folder
image_dir = os.path.join(DIR, "training_images")

# to get the region of interest of our training images
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# create our face recogniser
recogniser = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        # only interested in image files
        if file.endswith("pgn") or file.endswith("jpg"):
            # get the path of each image
            path = os.path.join(root, file)
            # get the label by getting the folder name before it e.g "caolan" for caolans images
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # replace just replaces any spaces with a dash
            #print(label, path)

            if not label in label_ids:
                # attach an id to that label
                # e.g caolan will have an id of 0
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            #print(label_ids)

            # convert image to greyscale, because thats how opencv recognises faces, with greyscale
            image = Image.open(path).convert("L")

            # convert the image into an array of pixels with the color value of grey (0-255)
            image_array = np.array(image, "uint8")

            # do face detection for the image
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                region_of_interest = image_array[y:y+h, x:x+w]
                # add to our training data
                x_train.append(region_of_interest)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as file: # wb is to write
    pickle.dump(label_ids, file)

# train our recogniser
recogniser.train(x_train, np.array(y_labels))
recogniser.save("trainer.yml")