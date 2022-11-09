import cv2
import face_recognition as fr

#cargar imagenes
foto_control = fr.load_image_file('FotoA.jpg')
foto_prueba = fr.load_image_file('FotoB.jpg')

#pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#localizar cara control
lugar_cara_A = fr.face_locations(foto_control)[0]
lugar_cara_B = fr.face_locations(foto_prueba)[0]

#codificar la cara
cara_codificada_A = fr.face_encodings(foto_control)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

#permite q el usuario lo pueda ver
#print(lugar_cara_A) me da las coordenadas donde se encuentra el rostro en la imagen - SENTIDO RELOJ
cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0,255,0), # color rgb
              2) #borde

cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0,0,255), # color rgb
              2) #borde

#mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto prueba', foto_prueba)

#mantener el programa abierto
cv2.waitKey(0)
