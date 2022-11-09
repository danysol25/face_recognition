import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

#le pido al programa que encuentre mi carpeta
#crear BD
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta) # toma el nombre con el que la imagen fue guardado (.jpg incluido)

print(lista_empleados)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])
print(nombres_empleados)

#codificar imágenes
def codificar(imagenes):
    #crear lista nueva
    lista_codificada = []
    
    #pasar las imagenes a RGB
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        #encontrar la cara en la imagen
        codificado = fr.face_encodings(imagen)[0]

        #agregar a la lista
        lista_codificada.append(codificado)

        #devolver lista codif
    return lista_codificada

#registrar los ingresos
def registrar_ingresos(persona):
    f=open('registro.csv', 'r+') #r+ para no sólo abrirlo, sino escribirlo
    lista_datos = f.readlines() #lee todas las líneas de nuestro archivo
    nombres_registro = [] #aquí registraremos a las personas
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0]) 

    if persona not in nombres_registro: #quiere decir q todavía no ha ingresado al trabajo
        hora = datetime.now()
        string_ahora = hora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona}, {string_ahora}')



lista_empleados_codificada = codificar(mis_imagenes) #creo una nueva variable, que almacena mi función, y de parámetro le doy la OTRA variable donde guardé mis imagenes
print(len(lista_empleados_codificada))

#tomar una imagen de cámara web
captura = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#leer la imagen de la cámara
exito, imagen = captura.read()

if not exito: #si no se ha podido tomar la foto
    print('No se ha podido tomar la captura')
else:
    cara_captura = fr.face_locations(imagen) #localiza la cara dentro de la imagen
    #codificar la cara
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    #buscar coincidencia entre la cara y la lista de empleados
    for caracdif, caraubic in zip(cara_captura_codificada, cara_captura):
        #toma una a una las ubicaciones que haya en la lista cara_captura, usamos ZIP para hacerlo en el mismo loop
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracdif)
        distancias = fr.face_distance(lista_empleados_codificada, caracdif)
        #la distancia menor (el rostro guardado vs capturado q más coincidencia tenga), es el empleado que será reconocido
        print(distancias)

        indice_coincidencia = numpy.argmin(distancias)

        #mostrar coincidencias
        
        if distancias[indice_coincidencia] > 0.6:
            print('No coincide con ningún empleado')
        else:
            nombre = nombres_empleados[indice_coincidencia]

            registrar_ingresos(nombre)
            
            #mostrar la imagen de la cámara web
            cv2.imshow('Imagen web', imagen)
            #mantener ventana abierta
            cv2.waitKey(0)
            print(f'Bienvenidx {nombre}')
