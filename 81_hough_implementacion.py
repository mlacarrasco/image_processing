"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Procesamiento Digital de Imágenes
 Miguel Carrasco (miguel.carrasco@uai.cl)

 Ejemplo: proceso de acumulador de transformada de Hough
 %rev.$1.0.$ date: 10/06/2024
 %rev.$1.1.$ date: 13/06/2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import peak_local_max

def mapa_acumulacion(bordes,min_distance,threshold):
    #funcion que determina el mapa de acumulacion según los bordes
    #distancia de diagonal de la imagen (distancia máxima)
    diagonal = np.sqrt(bordes.shape[0]**2 + bordes.shape[1]**2)

    # calculamos un arreglo con distancias desde el inicio de la imagen
    # hacia las diagonales. De este modo podemos tener distancias con ángulos
    # negativos
    rho_step = 12  #define el paso de puntos de distancia a considerar
    rho_distancia = np.arange(-diagonal, diagonal, rho_step)

    #matriz de posiciones (x,y) de los pixeles del borde la región
    x, y = np.where((bordes==255))

    # angulos sobre los cuales vamos a operar la ecuación
    no_angles = 360
    theta = np.linspace(-np.pi/2, np.pi/2, no_angles)

    #inicializamos la matriz de acumulación de RHO Y ANGULOS
    acumulador = np.zeros((len(rho_distancia), no_angles))

    # vamos a evaluar esta ecuación para cada angulo theta
    # rho = x * np.cos(theta) + y * np.sin(theta) 
    rho = lambda theta : x * np.cos(theta) + y * np.sin(theta)

    #aplicamos la función map para iterar sobre el arreglo theta
    # esto nos permite obtener una matriz con distancias 
    # (cada fila un angulo y cada columna una cada coordenada [x,y] del borde)
    mapa_de_distancias = map(rho, theta) 

    # la funcion min_row retorna el índice (o distancia) entre un valor
    # y la distancia definida en rho_distancia
    min_row = lambda val: np.argmin(np.abs(val-rho_distancia))
            
    # vamos a recorrer la lista con distancia. 
    # Cada fila representa un ángulo
    for theta_ang, row in enumerate(list(mapa_de_distancias)):
        
        #recorremos cada vector de distancias según cada angulo del ciclo for
        #y buscamos la posición con la menor distancia
        id_minimo = list(map(min_row, row))
        
        # incrementamos el contador según cada mínimo encontrado
        # previamente. Este proceso se realiza para cada punto (x,y)
        for i in id_minimo:
            acumulador[i, theta_ang]+=1
    
    # buscamos los peaks del mapa de acumulación
    # esto retorna una coordenada x,y del mapa de acumulación
    xy = peak_local_max(acumulador, min_distance=min_distance, threshold_abs=threshold)
    x,y = zip(*xy)
    
    #buscamos los ángulos y rho a partir de las coordenadas
    ang_theta = theta[list(y)]
    rho_dist = rho_distancia[list(x)]
    
    #empaquetamos los resultados obtenidos
    valores_maximos = [ang_theta, rho_dist]
    
    
    return acumulador, valores_maximos, xy
    

# ***************************************
#           PROGRAMA PRINCIPAL          *
# ***************************************

if __name__=='__main__':
    #programa ppal
    min_distance = 11
    threshold = 5
    
    I = cv2.imread('imagenes/rombo.png', cv2.IMREAD_GRAYSCALE)
    bordes = cv2.Canny(I,10,150,apertureSize = 3)

    #estimamos el mapa de acumulación
    acum,valores_maximos, xy = mapa_acumulacion(bordes,
                                                min_distance, 
                                                threshold)
    
    plt.figure()
    plt.imshow(acum, cmap='hot')
    plt.scatter(xy[:,1], xy[:,0],s=20)
    plt.title('mapa de acumulación')
    plt.show()

    fig = plt.figure()
    plt.imshow(I, cmap='gray')
    eje_x = np.array((0, bordes.shape[1]))
    
    for theta, rho in zip(*valores_maximos):  #el * separa un arreglo
        y0 = (rho - eje_x[0] * np.cos(theta)) / np.sin(theta)
        y1 = (rho - eje_x[1] * np.cos(theta)) / np.sin(theta)
        plt.plot((y0, y1), eje_x, '-r')
    plt.xlim(0,bordes.shape[1])
    plt.ylim(0,bordes.shape[0])
    plt.show()
    
  



