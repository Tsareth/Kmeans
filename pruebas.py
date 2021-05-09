import numpy as np                          # Importar librerias necesarias
import cv2
import keyboard
import imutils
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    
    video = cv2.VideoCapture("Video.mp4");
    width = video.get(cv2. CAP_PROP_FRAME_WIDTH )
    height = video.get(cv2. CAP_PROP_FRAME_HEIGHT )
    
    while(True):                                                                #Revisa si el video se encuentra abierto       
        ret, frame = video.read()                                #Obtiene la informacion del video
        if ret == True:
            frame_resized = cv2.resize(frame,(int(width/2),int(height/2)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        else:
            break
        if frame is None:
            break
        start = time.time()
        kmeans(frame_resized, width,height)
        end = time.time()
        cv2.imshow('Resultado',frame)                            # Abre en una pantalla el video procesado
        print("Elapsed (with compilation) = %s" % (end - start))
        k = cv2.waitKey(30) & 0xff                              # Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                  # Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break
    cv2.destroyAllWindows()                                     # Se destruyen las pantallas creadas por la aplicacion
    print('Borrado exitoso')


def nClusters(frame):
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    img=img.reshape((img.shape[1]*img.shape[0],3))
    md=[]
    for i in range(1,15):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(img)
        o=kmeans.inertia_
        md.append(o)

    plt.plot(list(np.arange(1,15)),md)
    plt.show()

def kmeans(frame, width_original, height_original):
    nClusters(frame)
    
    Z = frame.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 9, 1.0)
    K = 8
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    #k = cv2.waitKey(50) & 0xff
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    frame_resized = cv2.resize(res2,(int(width_original),int(height_original)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('res2',frame_resized)

main()
