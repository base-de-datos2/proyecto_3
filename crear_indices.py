import face_recognition
import numpy as np
import os
from utility_functions import ImgObject
import faiss



with open('vector_imgs.npy',"rb") as file:
    imgs_encodings = np.load(file,allow_pickle=True)

faiss_index= faiss.IndexHNSWFlat(128,32)

for i in range(len(imgs_encodings)):
    local_img = imgs_encodings[i]
    faiss_index.add(np.array([local_img.file_img_encoding],np.float32))

faiss.write_index(faiss_index,"faiss_index.dat")




