import numpy as np                          # Importar librerias necesarias
import cv2
import keyboard
import imutils
import time


# Definicion de variables globales

global k
k = 8
global name_k
global name
global width
global height
global fps
global resolucion
resolucion = 2
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    
    global name                                                                         #Se hace un llamado a las variables globales que se desean modificar 
    global width
    global height
    global fps
    global k
    
    name = input('Indique el nombre del video en la carpeta que desea analizar')        #Input del usuario paara saber el video a buscar, este debe ser un path completo o un nombre si
                                                                                        #Se encuentra en el mismo directorio que el codigo
    k = int(input('Indique la cantidad de centros que desea para kmeans'))              #Se pregunta al usuario cual valor de k desea utilizar para kmeans
    video = cv2.VideoCapture(name);                                                     #Se abre una instancia del video original
                                                                                        #Se obtiene informacion general del video original y se asigna a las variables globales 
    width = video.get(cv2. CAP_PROP_FRAME_WIDTH )
    height = video.get(cv2. CAP_PROP_FRAME_HEIGHT )
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = count_frames(video, override=False)
    duracion = total_frames/fps                             



    opcion = MenuOpciones(width,height,fps,total_frames,duracion)                       #Se llama a la funcion que despliega el menu de opciones de la aplicacion
    
    CheckForKmeans()                                                                    #Se llama a la funcion que revisa si ya existe un video procesado
    
    if (opcion == 1):                              
        PorFrames(total_frames)                                                         #Se hace llamada a la funcion que permite ver los videos por frames
     
    if(opcion == 2):
        Continuo()                                                                      #Se hace llamada a la funcion que permite ver los videos continuos

        
    video.release()                                                                     #Se elimina esta instancia del video original
    cv2.destroyAllWindows()                                                             #Se destruyen las ventanas existentes
    print('Borrado exitoso')                                                            


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion de conteo automatico de frames de la libreria de openCV

def count_frames(video, override=False):
    
    total = 0
    if (override):                                                                      #Si el override se indica en true, se pasa al conteo manual
        total = count_frames_manual(video)
    else:
        try:                                                                            #Si el override es false, se obtiene el valor a partir de la funcion de openCV
    
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = count_frames_manual(video)                                          #Si el conteo con la funcion de openCV falla, se realiza el conteo manual
    
    return total


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion de conteo manual de frames

def count_frames_manual(video):                                                         #Metodo manual de conteo de frames del video
    
    total = 0
    
    while (True):                                                                       #Ciclo que realiza una suma a una variable cada vez que un frame nuevo es leido
        
        (grabbed, frame) = video.read()
        
        if not grabbed:                                                                 #Si existe un error con la lectura o ya no hay mas frames por leer, se cierra el ciclo
            
            break
        
        total += 1
        
    return total

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion de transformacion de un frame a su version de kmeans

def kmeans(frame, width_original, height_original):

    global k                                                                            #Variable global que representa la cantidad de centros del kmeans
    
    Z = frame.reshape((-1,3))                                                           #El frame a analizar se vuelve una matriz con la cantidad de pixeles y sus canales de color
    Z = np.float32(Z)                                                                   #Se asegura que los valroes de la matriz sean tipo float para su manipulacion mediante la 
                                                                                        #libreria numpy
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)            #Se definen los criterios a utilizar con openCV para el uso de cv2.kmeans()
                                                                                        #Donde cv2.TERM_CRITERIA_EPS se refiere al erro valido por los datos y 
                                                                                        #cv2.TERM_CRITERIA_MAX_ITER la cantidad de iteraciones maxima
    
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)           #funcion de openCV que genera clusters tipo kmeans de los datos indicados
                                                                                        #Los datos se presentan en Z, k es la cantidad de centros, 10 se refiere a la cantidad de
                                                                                        #Intentos que la funcion va a ejecutarse para buscar el mejor resultado y
                                                                                        #cv2.KMEANS_PP_CENTERS se refiere al metodo de seleccion de los centros, en este caso busca
                                                                                        #por un metodo probabilistico los mejores centros pero aumenta el costo computacional
    
    center = np.uint8(center)                                                           #Se transforman los centros a enteros para porceder a obtener la nueva imagen
    
    res = center[label.flatten()]                                                       #Se vuelve a crear la imagen 2d y se le da las mismas caracteristicas que el frame original
    res2 = res.reshape((frame.shape))
    
    frame_resized = cv2.resize(res2,
                               (int(width_original),int(height_original)),
                               fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    return frame_resized
    


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion que realiza el procesado de todo el video para luego presentarlo al usuario. Unicamente es llamado si el procesado no se ha realizado anteriormente.
    
def preprocessing(video, frames_kmeans, width, height):
    global resolucion
    while (True):                                                                             
        ret, frame = video.read()                                                       #Video.read() busca el frame siguiente del array y devuelve los datos frame y ret
                                                                                        #Ret es un booleano que estrue si la lectura del video fue exitosa
        if ret == True:                                                                 #Se revisa si la lectura del video es exitosa y cambia la resolucion del frame a la mitad del
                                                                                        #original para asi reducir el costo computacional de la operacion kmeans
            frame_resized = cv2.resize(frame,
                                       (int(width/resolucion),int(height/resolucion)), 
                                       fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        else:                                                                           #Si existe un error en la lectura del video, se cierra el ciclo
            break
        if frame is None:                                                               #Si al leer el proximo frame del video este es nulo, se cierra el ciclo
            break
        
        frames_kmeans.append(kmeans(frame_resized, width,height))                       #Se crea una lista de frames creados por la funcion kmeans para crear el nuevo video resultado
        
        k = cv2.waitKey(30) & 0xff                                                      # Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:                                                                     #Al presionar la tecla esc o q, el ciclo se cierra y termina la ejecucion del programa
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                              
            break
    return frames_kmeans


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion que imprime las opciones de esta aplicacion


def MenuOpciones(width, height,fps,total_frames,duracion):
    
    print("---------------Analisis de video por K-means-------------------")

    print("El video elegido presenta las siguientes caracteristicas")
    print("Ancho = %s " % (width))
    print("Alto = %s " % (height))
    print("Fps = %s" % (fps))
    print("Duracion total (segundos) = %s" % (duracion))
    print("Cantidad de frames = %s" % (total_frames))
    print("Menu de opciones")
    print("1. Analisis de video por frames")
    print("           Mediante las flechas del teclado puede avanzar un frame a la vez para observar el analsis de k-means")
    print("2. Analisis de video continuo")
    print("           El analisis de k-means se realiza de forma automatica y se presentan los frames continuos")
    print("3. Mas informacion de la aplicacion")
    opcion = int(input())

    if opcion == 3:
        print("Aplicacion creada por Armando Uribe Castro y Giancarlo Vargas Villegas")
        print("Entrega de la tarea 2 de segmentacion para el curso de Visión del ITCR")
        print("Este sistema cuenta con una funcion de preprocesado que se ejecuta cuando un video no cuenta con su version de kmeans")
        print("Esta funcion crea un video con todos los frames procesados")
        print("Al finalizar el procesado se abren las ventanas con ambos videos")
        print("Este comportamiento permite evitar procesar un video multiples veces y brinda un menor tiempo de ejecucion")
        print("Modo por frames:")
        print("Al seleccionar las flechas de izquierda y derecha del teclado, se realiza un salto al frame anterior y siguiente")
        print("Al seleccionar la tecla escape se detiene el proceso y se termina la ejecucion de la aplicacion")
        print("Continuo:")
        print("Este modo carga los videos correspondientes y los presenta de manera continua a velocidad normal")
        print("Al seleccionar la tecla escape se detiene el proceso y se termina la ejecucion de la aplicacion")
        MenuOpciones(width, height,fps,total_frames,duracion)

    if opcion > 3:
        print('Por favor elija una opcion valida')
        print('')
        MenuOpciones(width, height,fps,total_frames,duracion)

    return opcion

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion que revisa el directorio en el que el codigo se encuentra y busca un video del procesado realizado en una ejecucion pasada

def CheckForKmeans():
    global name                                                                         #Se llama a la variable global name del nombre del video
    
    video = cv2.VideoCapture(name)                                                      #Se crea una instancia del video original en esta funcion
    frames_kmeans = []                                                                  #Se crea la variable de frames del resultado luego de procesar para obtener los frames con kmeans
    global name_k
    global resolucion
    name_k = "Resultado_"+"Res"+ str(resolucion) +"K"+ str(k) + name                                                #Se modifica el valor del nombre del video ya procesado realizado en una ejecucion pasada
    video_kmeans = cv2.VideoCapture(name_k)                                             #Se crea una instancia del video ya procesado en esta funcion
    ret2, frameK = video_kmeans.read()                                                  #Se lee el primer frame de este video para reconocer si el video existe en el directorio
    if (frameK is None):                                                                #Si existe un problema al leer este frame se procede a realizar el procesado para el video
                                                                                        #seleccionado
        video_kmeans.release()                                                          #Se elimina la instancia del video procesado que se intento abrir
        print('El video no presenta resultado creado previamente')
        print('Se procedera a crear el kmeans para cada frame')
        print('Esto prodria tardar un minuto')
        start = time.time()                                                             #Se inicio el conteo del tiempo para el procesado del video
        frames_kmeans = preprocessing(video,frames_kmeans, width, height)               #Se inicia el procesado del cada frame individual
        end = time.time()                                                               #Se termina el conteo del tiempo para el procesado del video
        video_kmeans = cv2.VideoWriter(name_k,                                          #Se inicializa un creador de videos para crear el video procesado
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       int(fps),
                                       (int(width),int(height)),
                                       True)
        print("Elapsed (with compilation) = %s" % (end - start))
        for i in range(len(frames_kmeans)):                                             #Se escribe cada frame obtenido en el procesado en este nuevo video 
            video_kmeans.write(frames_kmeans[i])
            
        video_kmeans.release()                                                          #Se libera el video para luego llamarlo como otro tipo de variable
        video_kmeans = cv2.VideoCapture(name_k)                                         #Se llama la instancia de este video como captura de openCV

    else:                                                                               #Si la lectura del primer frame fue correcto se abre una instancia del video ya procesado
        video_kmeans = cv2.VideoCapture(name_k);
        video_kmeans.release()
        #total_framesK = count_frames(video_kmeans, override=True)
        #print(total_framesK)
        print('Se ha encontrado un procesado previamente creado')
        print('Se abrira esta version para mayor rapidez')
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Funcion que permite la seleccion del frame por medio de teclas del keyboard mediante la libreria keyboard

def PorFrames(total_frames):        
    video = cv2.VideoCapture(name);                                                     #Se abre el video original como captura                                                                    
    video_kmeans = cv2.VideoCapture(name_k);                                            #Se abre el video procesado como captura
    ret, frame = video.read()                                                           #Se lee el primer frame del video original
    ret2,frameK = video_kmeans.read()                                                   #Se lee el primer frame del video procesado
    cv2.imshow('frame', frame)                                                          #Se enseña en pantalla el primer frame del video original
    cv2.imshow('K-means',frameK)                                                        #Se enseña en pantalla el primer frame del video procesado
    frame_actual = 0                                                                    #Se declara el frame actual como 0 para el inicial.
    while(True):
        k = cv2.waitKey(30) & 0xff                                                      # Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones 
        if k == 27:
            break
        if keyboard.is_pressed('left'):                                                 #Se revisa si el evento de presionar la tecla izquierda sucedio
            if frame_actual > 0:                                                        #Se revisa si el frame actual es mayor que cero para evitar asignaciones de posicion negativa
                frame_actual = frame_actual - 1                                         #Si el frame es mayor que 0, es posible pasar al frame anterior, por lo que se resta 1 a la 
                                                                                        #variable
                video.set(1,frame_actual)                                               #Se asigna el frame anterior para la lectura siguiente
                video_kmeans.set(1,frame_actual)                                        #Se asigna el frame anterior para la lectura siguiente
                ret, frame = video.read()                                               #Se realiza la lectura del frame para desplegar en pantalla
                ret2,frameK = video_kmeans.read()                                       #Se realiza la lectura del frame para desplegar en pantalla
                
                cv2.imshow('frame', frame)                                              #Se enseñan los frames en pantalla del video original y procesado
                cv2.imshow('K-means', frameK)
                
            else:
                frame_actual = frame_actual                                             #Si el frame actual es el primero, no se realizan cambios al estripar la tecla izquierda
        
        if keyboard.is_pressed('right'):                                                #Se revisa si el evento de presionar la tecla izquierda sucedio
            if frame_actual < total_frames-1:                                           #Se revisa si el frame actual es mayor que cero para evitar asignaciones de posicion negativa
                frame_actual = frame_actual + 1                                         #Si el frame es mayor que 0, es posible pasar al frame anterior, por lo que se resta 1 a la
                                                                                        #variable
                video.set(1,frame_actual)                                               #Se asigna el frame anterior para la lectura siguiente
                video_kmeans.set(1,frame_actual)                                        #Se asigna el frame anterior para la lectura siguiente
                ret, frame = video.read()                                               #Se realiza la lectura del frame para desplegar en pantalla
                ret2,frameK = video_kmeans.read()                                       #Se realiza la lectura del frame para desplegar en pantalla
                
                cv2.imshow('frame', frame)                                              #Se enseñan los frames en pantalla del video original y procesado
                cv2.imshow('K-means', frameK)
                
            else:
                frame_actual = frame_actual                                             #Si el frame actual es el ultimo, no se realizan cambios al estripar la tecla derecha

        if keyboard.is_pressed('esc'):                                                  #Si la tecla escape se estripa, se cierra el ciclo                                          
            frame = 0
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                                          #Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break
    video.release()                                                                     #Se liberan los videos de esta instancia
    video_kmeans.release()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def Continuo():    
    video = cv2.VideoCapture(name);                                                     #Abre los videos original y procesado en esta instancia
    video_kmeans = cv2.VideoCapture(name_k);
    while(True):                                                                              
        ret, frame = video.read()                                                       #Obtiene la informacion del video original
        ret2,frameK = video_kmeans.read()                                               #Obtiene la informacion del video procesado
        
        if frame is None:                                                               #Si se lee el frame y este es nulo (se llega al ultimo), se cierra el ciclo
            break
        
        cv2.imshow('Video',frame)                                                       #Abre en una pantalla el video procesado y el original
        cv2.imshow('K-means', frameK)
        

        k = cv2.waitKey(30) & 0xff                                                      #Crea un delay de 30 milisegundos para que la pantalla sea capaz de realizar las operaciones
                                                                                        #Si se estripa la tecla esc se cierra el ciclo
        if k == 27:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):                                          #Si la tecla q es presionada, se detiene la presentacion de la mascara creada
            break
    video.release()                                                                     #Se cierran las instancias del video original y procesado de esta funcion
    video_kmeans.release()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

main()
