import cv2
import face_recognition as fr

#cargar imagenes
foto_control = fr.load_image_file('FotoA.jpg')
foto_prueba = fr.load_image_file('FotoB.jpg')

#pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto prueba', foto_prueba)

#mantener el programa abierto
cv2.waitKey(0)