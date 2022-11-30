import numpy as np
import cv2
import sympy as sym  # Para cálculo simbolico y detalles esteticos
from scipy.special import factorial  # Versión vectorizada de n! = n×(n-1)!
import matplotlib.pyplot as plt  # Para el gráfico de las funciones
from matplotlib.colors import Normalize, Colormap  # Para el coloreo dinamico
from matplotlib import ticker  # Detalles para ejes/barras de color
from matplotlib import rcParams  # Para aumentar la resolución de los gráficos
from tkinter import *
from tkinter import ttk, Entry, Button, Label, Tk, Text, StringVar
from tkinter import messagebox
import tkinter as tk
import tkinter.filedialog
from PIL import Image


img = cv2.imread('F.jpg')
cv2.imwrite("C:/Users/52221/Desktop/imagenesdigitales/imgoriginal.png", img)


x = y = 600
img = cv2.resize(img, (x, y))

# GRISES
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = np.float64(gray)

# TRANSFORMADA DE FOURIER EN 2D
frr = np.fft.fft2(gray2)
frr = np.fft.fftshift(frr)

# CALCULAR LA MAGNITUD DEL ARREGLO
frr_abs = np.abs(frr)

# ESPECTRO DE FRECUENCIA EN ESCALA LOGARITMICA
frr_log = 20*np.log10(frr_abs)

# MOSTRAMOS LA IMAGEN
cv2.imshow("Imagen Original", img)
img_frr = np.uint8(255*frr_log/np.max(frr_log))
cv2.imshow("Espectro de Fourier Logaritmica", img_frr)
cv2.imwrite(
    "C:/Users/52221/Desktop/imagenesdigitales/EspectrodeFourierLogaritmica.png", img_frr)

# FILTRO PASA ALTO
# Parte central valores cercanos al cero.
# y el resto de valores sean altos
F1 = np.arange(-x/2+1, x/2+1, 1)
F2 = np.arange(-y/2+1, y/2+1, 1)
[X, Y] = np.meshgrid(F1, F2)    # arreglo matricial de las combinaciones
D = np.sqrt(X**2+Y**2)    # distancia del centro
D = D/np.max(D)
# DEFINIR RADIO DE CORTE
Do = 0.10
# Creación del Filtro Ideal en 2D
Huv = np.zeros((x, y))  # matriz de ceros
# PRIMERO CREAR EL FILTRO PASA BAJO IDEAL
for i in range(x):
    for j in range(y):
        if (D[i, j] < Do):
            Huv[i, j] = 1
# CONVERTIR A PASA ALTO IDEAL
Huv = 1-Huv

# ----------------------------------------------------
cv2.imshow("FILTRO 2D PASA ALTO IDEAL", np.uint8(255*Huv))
cv2.imwrite(
    "C:/Users/52221/Desktop/imagenesdigitales/FILTRO2DPASAALTOIDEAL.png", Huv)


# --------------------------FILTRADO EN FRECUENCIA
# -MULTIPLICACIÓN ELEMENTO A ELEMENTO EN EL DOMINIO DE LA FRECUENCIA
Guv = Huv*frr
# MAGNITUD
Guv_abs = np.abs(Guv)
Guv_abs = np.uint8(255*Guv_abs/np.max(Guv_abs))
cv2.imshow('ESPECTRO DE FRECUENCIA G', Guv_abs)
cv2.imwrite(
    "C:/Users/52221/Desktop/imagenesdigitales/ESPECTRODEFRECUENCIAG.png", Guv_abs)

# ---TRANSFORMADA INVERSA PARA OBTENER LA SEÑAL FILTRADA
# IFFT2
gxy = np.fft.ifft2(Guv)
gxy = np.abs(gxy)
gxy = np.uint8(gxy)
# --MOSTRAR LA IMAGEN FILTRADA
cv2.imshow('IMAGEN FILTRADA', gxy)
cv2.imwrite(
    "C:/Users/52221/Desktop/imagenesdigitales/IMAGENFILTRADA.png", gxy)
#cv2.waitKey(0)
cv2.destroyAllWindows()

ventana = tk.Tk()
ventana.geometry("1200x800")
ventana.title('Transformada de Fourier en Imagebnes Digitales')

imagenoriginal = PhotoImage(file="imgoriginal.png")
imagenoriginalsub = imagenoriginal.subsample(2)
fondo1 = Label(ventana,image=imagenoriginalsub).place(x=10,y=100)

imagen1 = PhotoImage(file="EspectrodeFourierLogaritmica.png")
imagen1sub = imagen1.subsample(2)
fondo2 = Label(ventana,image=imagen1sub).place(x=200,y=100)

imagen2 = PhotoImage(file="FILTRO2DPASAALTOIDEAL.png")
imagen2sub = imagen2.subsample(2)
fondo3 = Label(ventana,image=imagen2sub).place(x=510,y=100)

imagen3 = PhotoImage(file='IMAGENFILTRADA.png')
imagen3sub = imagen3.subsample(2)
fondo4= Label(ventana,image=imagen3sub).place(x=510, y=410)

imagen4 = PhotoImage(file='ESPECTRODEFRECUENCIAG.png')
imagen4sub = imagen3.subsample(2)
fondo5= Label(ventana,image=imagen4sub).place(x=200, y=410)

tiutlo1 = Label(ventana,text='Imagen Original',fg='#c311b1',font=('Verdana',12)).place(x=20,y=75)
tiutlo2 = Label(ventana,text='Espectro de Fourier Logaritmica ',fg='#c311b1',font=('Verdana',12)).place(x=220,y=75)
tiutlo3 = Label(ventana,text='FILTRO 2D PASO ALTO IDEA',fg='#c311b1',font=('Verdana',12)).place(x=530,y=75)
tiutlo4 = Label(ventana,text='IMAGEN FILTRADAl',fg='#c311b1',font=('Verdana',12)).place(x=530,y=713)
tiutlo5 = Label(ventana,text='ESPECTRO DE FRECUENCIA G',fg='#c311b1',font=('Verdana',12)).place(x=220,y=713)

ventana.mainloop()