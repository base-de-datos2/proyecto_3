import face_recognition
import numpy as np
import os
from utility_functions import ImgObject
import faiss
from rtree import index



with open('vector_imgs.npy',"rb") as file:
    imgs_encodings = np.load(file,allow_pickle=True)

# faiss_index= faiss.IndexHNSWFlat(128,32)

# for i in range(len(imgs_encodings)):
#     local_img = imgs_encodings[i]
#     faiss_index.add(np.array([local_img.file_img_encoding],np.float32))

# faiss.write_index(faiss_index,"faiss_index.dat")

p = index.Property()
p.dimension = 128
p.dat_extension = "data"
p.idx_extension = "index"

idx = index.Index("puntos", properties = p)

for i in range(len(imgs_encodings)):
    local_img = imgs_encodings[i]
    img = local_img.file_img_encoding.tolist()
    print(i)
    idx.add(i,img)

result = idx.nearest(imgs_encodings[1234].file_img_encoding.tolist(),num_results=5)

for i in result:
    print(i)


idx.close()


