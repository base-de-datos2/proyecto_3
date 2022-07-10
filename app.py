from venv import create
from flask import Flask, render_template, jsonify,request,redirect
import psycopg2
from PIL import Image
import time
import face_recognition



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/range-query')
def range_query():
    return render_template('range_query.html')



@app.route('/knn-query',methods=['GET','POST'])
def knn_query():
    return render_template('knn_query.html')
   



@app.route('/make-knn-query',methods=['POST'])
def query():
    print(request)
    print(request.files)
    # text = request.args.get('query')
    # k = request.args.get('k')
    # start = time.time()
    # keys, dict = create_tf_query(procesamiento(text))
    # create_unit_vector_query('sorted_tokens.txt', 2507 -1 , keys, dict)
    # tweets = topk(2507 -1, int(k))

    # tiempo_spimi = time.time() - start


    # return render_template('topk.html', tweets = tweets, tweets_postgres = query_postgres_result, tiempo_postgres = tiempo_postgres, tiempo_spimi = tiempo_spimi)
    return render_template('display_knn_query.html') 


if __name__ == '__main__':
    app.run('0.0.0.0', port = 1042  , debug = True)