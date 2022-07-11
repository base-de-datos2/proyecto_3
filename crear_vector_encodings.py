import face_recognition
import numpy as np
import os
from utility_functions import ImgObject
import faiss



directories = os.listdir("./lfw/") 
imgs_encodings = np.array([],dtype=ImgObject) 

count = 0
for dir in directories:

    dir_files = os.listdir("./lfw/" + dir)
    last_encoding = None
    for file in dir_files: 
        path = "./lfw/" + dir + "/" + file
        file_img = face_recognition.load_image_file(path )
        file_img_encoding = face_recognition.face_encodings(file_img)
        if(len(file_img_encoding) != 0):
            images_encodings = np.append(images_encodings,[ImgObject(dir,path,file_img_encoding[0])])
            count += 1

with open('vector_imgs.npy','wb') as file:
    np.save(file,images_encodings,allow_pickle=True)


