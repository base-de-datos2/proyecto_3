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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/range-query')
def range_query():
    return render_template('range_query.html')



@app.route('/knn-query',methods=['GET','POST'])
def knn_query():
    return render_template('knn_query.html')
   





@app.route('/get-img/<filename>')
def send_img(filename):
    filename = filename.replace('}','/')
    return send_file(filename)


@app.route('/make-knn-query',methods=['POST'])
def query():
    print(request)
    file = request.files['file']
    img = face_recognition.load_image_file(file)
    img_encoding = face_recognition.face_encodings(img)
    _,I = faiss_index.search(np.array([img_encoding[0]],np.float32),int(request.form['k']))
    faiss_results = []

    for result in I[0]:
        path = imgs_encodings[result].path[2:].replace('/','}')
        # print(path)
        faiss_results.append(path)


    # Rtree
    p = rtindex.Property()
    p.dimension = 128
    p.dat_extension = "data"
    p.idx_extension = "index"

    idx = rtindex.Index("puntos", properties = p)

    rtree_knn = idx.nearest(img_encoding[0], num_results = int(request.form['k']))
    rtree_results = []

    for result in rtree_knn:
        path = imgs_encodings[result].path[2:].replace('/', '}')
        rtree_results.append(path)


    # Sequential
    top_k = PriorityQueue()

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
        print(top)
        sequential_index = top[1]
        path = imgs_encodings[sequential_index].path[2:].replace('/', '}')
        sequential_results = [path] + sequential_results
        top_k.get()

    return render_template('display_knn_query.html',faiss_imgs_paths = faiss_results, sequential_imgs_paths = sequential_results, rtree_imgs_paths = rtree_results) 



if __name__ == '__main__':
    

    app.run('0.0.0.0', port = 1042  , debug = True)