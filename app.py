from venv import create
from flask import Flask, render_template, jsonify,request,redirect, send_from_directory,send_file
import time
import face_recognition
from utility_functions import ImgObject
import numpy as np
from queue import PriorityQueue
from rtree import index as rtindex
import faiss



app = Flask(__name__)

with open('vector_imgs.npy',"rb") as file:
        imgs_encodings = np.load(file,allow_pickle=True)

faiss_index= faiss.read_index("faiss_index.dat")


p = rtindex.Property()
p.dimension = 128
p.dat_extension = "data"
p.idx_extension = "index"


rtree_index = rtindex.Index("puntos", properties = p)





@app.route('/',methods=['GET','POST'])
def knn_query():
    return render_template('index.html')
   




@app.route('/get-img/<filename>')
def send_img(filename):
    filename = filename.replace('>','/')
    # print(filename)
    return send_file(filename)
    # return ""



@app.route('/make-knn-query',methods=['POST'])
def query():
    print(request)
    file = request.files['file']
    img = face_recognition.load_image_file(file)
    img_encoding = face_recognition.face_encodings(img)


    #Faiss

    faiss_time = time.time() 
    _,I = faiss_index.search(np.array([img_encoding[0]],np.float32),int(request.form['k']))
    faiss_time = time.time() - faiss_time
    faiss_results = []
    for result in I[0]:
        path = imgs_encodings[result].path[2:].replace('/','>')
        name = imgs_encodings[result].dir.replace('_',' ')
        faiss_results.append([path,name])


    # Rtree
   
    rtree_time = time.time()
    rtree_knn = rtree_index.nearest(img_encoding[0], num_results = int(request.form['k']))
    rtree_time = time.time() - rtree_time
    rtree_results = []
    

    for result in rtree_knn:
        path = imgs_encodings[result].path[2:].replace('/', '>')
        rtree_results.append([path,name])
        


    # Sequential

    top_k = PriorityQueue()

    sequential_time = time.time()
    for i in range(len(imgs_encodings) - 1):
        curr_encoding = imgs_encodings[i].file_img_encoding

        dist = face_recognition.face_distance([curr_encoding], img_encoding[0])

        if len(top_k.queue) < int(request.form['k']):
            top_k.put((-1 * dist, i))
        else:
            top = top_k.queue[0]
            if -1 * dist > top[0]:
                top_k.get()
                top_k.put((-1 * dist, i))

    sequential_results = []
    for i in range(len(top_k.queue)):
        top = top_k.queue[0]
        # print(top)
        sequential_index = top[1]
        path = imgs_encodings[sequential_index].path[2:].replace('/', '>')
        sequential_results = [[path,name]] + sequential_results
        top_k.get()
    sequential_time = time.time() - sequential_time

    return render_template('display_knn_query.html',faiss_imgs_paths = faiss_results, sequential_imgs_paths = sequential_results, rtree_imgs_paths = rtree_results,sequential_time=sequential_time,rtree_time=rtree_time,faiss_time=faiss_time) 



if __name__ == '__main__':
    

    app.run('0.0.0.0', port = 1042  , debug = True)