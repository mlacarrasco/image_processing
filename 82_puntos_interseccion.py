import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def interseccion_lineas(lines):
    
    index = 0
    number = lines.shape[0]
    intersections = np.array(np.zeros([number*number,2]))
    for i in range(0, number):
        line1= lines[i]
        for j in range(i+1,number):
            line2= lines[j]  
            intersections[index]= intersection(line1, line2)
            index+=1

    return intersections, index


#programa ppal
I = cv2.imread('imagenes/rombo.png', cv2.IMREAD_GRAYSCALE)

BW = cv2.Canny(I,50,150,apertureSize = 3)

angulos = np.linspace(-np.pi/2, np.pi/2, 360)
h, theta, d = hough_line(BW, theta=angulos)

# Mapa de acumulación
fig = plt.figure()
plt.subplot(111)
plt.imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.hot,aspect=1/15)

fig = plt.figure()
plt.subplot(111)
plt.imshow(BW, cmap=cm.gray)
eje_x = np.array((0, BW.shape[1]))

valores_maximos= hough_line_peaks(h, theta, d)
#*hough_line_peaks(h, theta, d) desarma el arreglo
max_lines = np.size(valores_maximos,1)
lines=np.array(np.zeros([max_lines,2]))
index=0

for accum, theta, rho in zip(*valores_maximos):
    y0 = (rho - eje_x[0] * np.cos(theta)) / np.sin(theta)
    y1 = (rho - eje_x[1] * np.cos(theta)) / np.sin(theta)
    lines[index]= [rho, theta]
    index+=1
    plt.plot(eje_x, (y0, y1), '-r')

intersections, num_ins = interseccion_lineas(lines)

for i in range(0,num_ins):
    x, y = intersections[i]
    plt.plot(x,y, 'yo')


plt.xlim(eje_x)
plt.ylim((BW.shape[0], 0))
plt.title('Lineas de mapa de Hough')

plt.show()