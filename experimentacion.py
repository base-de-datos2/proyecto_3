from rtree import index as rtindex
import faiss
import numpy as np
from queue import PriorityQueue
import face_recognition
from utility_functions import ImgObject
import os
import random
import time

with open('vector_imgs.npy',"rb") as file:
        imgs_encodings = np.load(file,allow_pickle=True)

times = {}

query_img_encoding = imgs_encodings[random.randint(0, 13175)].file_img_encoding

for size in (100, 200, 400, 800, 1600, 3200, 6400, 12800):
    print(size)
    if os.path.exists("puntos2.data"):
        os.remove('puntos2.data')
    if os.path.exists('puntos2.index'):
        os.remove('puntos2.index')

    times[size] = {}
    rand_index = [random.randint(0, 13175) for i in range(size)]

    top_k = PriorityQueue()

    start = time.time()
    for index in rand_index:
        curr_encoding = imgs_encodings[index].file_img_encoding

        dist = face_recognition.face_distance([curr_encoding], query_img_encoding)

        if len(top_k.queue) < 8:
            top_k.put((-1 * dist, index))
        else:
            top = top_k.queue[0]
            if -1 * dist > top[0]:
                top_k.get()
                top_k.put((-1 * dist, index))
    times[size]['sequential'] = time.time() - start

    print('finished sequential')

    p = rtindex.Property()
    p.dimension = 128
    p.dat_extension = "data"
    p.idx_extension = "index"

    idx = rtindex.Index("puntos2", properties = p)
    for index in rand_index:
        local_img = imgs_encodings[index]
        img = local_img.file_img_encoding.tolist()
        idx.add(index,img)

    start = time.time()
    rtree_knn = idx.nearest(imgs_encodings[0].file_img_encoding, num_results = 8)
    times[size]['rtree'] = time.time() - start

    print('finished rtree')

    faiss_index = faiss.IndexHNSWFlat(128, 32)

    for index in rand_index:
        local_image = imgs_encodings[index]
        faiss_index.add(np.array([local_image.file_img_encoding], np.float32))

    #faiss.write_index(faiss_index, "exp/faiss_index.dat")

    start = time.time()
    _,I = faiss_index.search(np.array([query_img_encoding], np.float32), 8)
    times[size]['faiss'] = time.time() - start

    print('finished faiss')


print(times)