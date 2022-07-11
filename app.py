from venv import create
from flask import Flask, render_template, jsonify,request,redirect, send_from_directory,send_file
import psycopg2
import time
import face_recognition
from utility_functions import ImgObject
import numpy as np
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
    filename = filename.replace('-','/')
    # print(filename)
    return send_file("./" + filename)

@app.route('/make-knn-query',methods=['POST'])
def query():
    print(request)
    file = request.files['file']
    img = face_recognition.load_image_file(file)
    img_encoding = face_recognition.face_encodings(img)
    _,I = faiss_index.search(np.array([img_encoding[0]],np.float32),int(request.form['k']))
    faiss_results = []

    for result in I[0]:
        path = imgs_encodings[result].path[2:].replace('/','-')
        name = imgs_encodings[result].dir.replace('_',' ')
        faiss_results.append([path,name])



    return render_template('display_knn_query.html',faiss_imgs_paths = faiss_results) 


if __name__ == '__main__':
    

    app.run('0.0.0.0', port = 1042  , debug = True)