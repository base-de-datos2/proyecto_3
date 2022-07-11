# Proyecto 3


|Nombre|Participación|Nota
|-|-|-|
|Francisco Magot|KNN Secuencial, KNN R-Tree|100%|
|Eric Bracamonte|KNN Faiss, App en Flask|100%|

1. KNN Secuencial

Con respecto a la búsqueda de KNN secuencial, se ha iterado por todas las imágenes dentro del folder "lfw". Para cada una de estas imágenes, se ha utilizado su vector característico correspondiente (utilizando la función face_encodings) para luego calcular la distancia euclideana a la imágen de búsquda (utilizando la función face_distance).
Se ha utilizado un max-heap para guardar los k elementos más cercanos.

2. KNN R-Tree

Con respecto a la búsqueda de KNN con R-Tree, se ha utilizado una libreria rtree en python. Dado a que el vector característico que devuelve la función face_encodings tiene 128 valores, se ha configurado el rtree para que soporte estas 128 dimensiones. Para conseguir los k vecinos más cercanos a la imágen de búsqeuda, se utilizará la función de la librería de rtree llamada "nearest". 

Para construir el índice, se debe correr el archivo [crear_indices.py](/crear_indices.py). 

3. KNN Faiss



## Experimentación

A continuación los tiempos de experimentación para los algoritmos implementados. Considerar que se utilizó un valor de K = 8 sobre una colección de imágenes de tamaño n. Todos los tiempos están expresados en segundos.

|n|KNN Secuencial|KNN RTree|KNN Faiss|
|-|-|-|-|
|100|0.0022|0.0004|0.0035|
|200|0.0039|0.0021|0.0001|
|400|0.0069|0.0025|0.0079|
|800|0.0146|0.0049|0.0002|
|1600|0.0264|0.0085|0.0001|
|3200|0.0548|0.0169|0.0001|
|6400|0.0983|0.0295|0.0002|
|12800|0.1914|0.065|0.0002|
