print(' LISTA 3 - REO 3')
print('AVANÇOS CIENTÍFICOS EM GENÉTICA E MELHORAMENTO DE PLANTAS I')
print('VISÃO COMPUTACIONAL NO MELHORAMTENTO DE PLANTAS')
print('Ewerton Lelys Resende')
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# ALUNOS:
# CAROLINE MARCELA DA SILVA
# EWERTON LÉLYS RESENDE
# MARIANA ANDRADE DIAS
# THIAGO TAVARES BOTELHO



print('=-'*55)

print('EXERCÍCIO 01: ')
print('Selecione uma imagem a ser utilizada no trabalho prático e realize os '
      'seguintes processos utilizando as bibliotecas OPENCV e Scikit-Image do Python:' )
print('-'*55)

#Importar pacotes
import cv2 # Importa o pacote opencv
import numpy as np
from matplotlib import pyplot as plt # Importa o pacote matplotlib
from skimage.measure import label, regionprops #pacote usado para trabalhar com imagens
from skimage.feature import peak_local_max  # IMPORTAR FUNÇOES ESPECICICAS
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd

# Leitura da imagem

nome_arquivo = 'trabalho.png'
img_bgr = cv2.imread(nome_arquivo,1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

print('a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare '
      'os resultados com a imagem original;  ')

# Filtros de médias
# Média com diferentes valores de KERNEL
img_fm_1 = cv2.blur(img_rgb,(3,3))

img_fm_2 = cv2.blur(img_rgb,(11,11))    # cv2.blur = é função que faz o fitro especificamente de médias
                                        # os tamanhos dos kernel se encontram entre parenteses.
img_fm_3 = cv2.blur(img_rgb,(21,21))    #quando aumentamos os valores aum. o numero de visinhos que estão sendo levados
                                        # em consideração para calcular a médias dos pixels que iram formar a imagem
                                        # os fitros são feitos para cada um dos canais (RGB)
                                        # os valore dos kelnels devem ser impares
img_fm_4 = cv2.blur(img_rgb,(31,31))

img_fm_5 = cv2.blur(img_rgb,(51,51))

# Apresentar imagens no matplotlib
plt.figure('Filtros de médias')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(2,3,2)
plt.imshow(img_fm_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3")

plt.subplot(2,3,3)
plt.imshow(img_fm_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("11x11")

plt.subplot(2,3,4)
plt.imshow(img_fm_3)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("21x21")

plt.subplot(2,3,5)
plt.imshow(img_fm_4)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("31x31")


plt.subplot(2,3,6)
plt.imshow(img_fm_5)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("51x51")

plt.show()

print('-'*55)
print('b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os'
      ' resultados entre si e com a imagem original. ')

# Diferentes tipos de Filtros

# Filtro de média
img_filtro_media1 = cv2.blur(img_rgb,(33,33)) # Faz uma média simple
img_filtro_media2 = cv2.blur(img_rgb,(61,61))

# Média ponderada
img_filtro_gaussiano1 = cv2.GaussianBlur(img_rgb,(33,33),0)# Média Ponderada, de forma que os vizinhos mais proximos tem peso maior
                                                          # 0 é o peso na média ponderada.
img_filtro_gaussiano2 = cv2.GaussianBlur(img_rgb,(61,61),0)

# Filtro de Mediana
 # faz a mediana dos valores dos pixes que estão na vizinhaca melhor de visualizar
img_filtro_mediana1 = cv2.medianBlur(img_rgb,33)
img_filtro_mediana2 = cv2.medianBlur(img_rgb,61)# 61 é tamanho do kernel

img_filtro_bilateral1 = cv2.bilateralFilter(img_rgb,33,33,21)
img_filtro_bilateral2 = cv2.bilateralFilter(img_rgb,61,61,21) # não perde os detalhismo das bordas, usando  2 funcoes glauciana
                                                             # 61 tamanho dos kernel
                                                             # 61 para intensidade
                                                             # 21 vai determinar a ponderação

#Filtros - kernel 31x31
plt.figure('Imagens- Diferentes filtros')
plt.subplot(231)
plt.imshow(img_rgb)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(img_filtro_media1)
plt.title("MÉDIA 33x33")
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(img_filtro_gaussiano1)
plt.title("GAUSSIANO 33x33")
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(img_filtro_mediana1)
plt.title("MEDIANA 33X33")
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.imshow(img_filtro_bilateral1)
plt.title("BILATERAL 33X33")
plt.xticks([])
plt.yticks([])

plt.show()

#Filtros - kernel 61x61
plt.figure('Imagens- Diferentes filtros')
plt.subplot(231)
plt.imshow(img_rgb)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(img_filtro_media2)
plt.title("MÉDIA 61X61")
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(img_filtro_gaussiano2)
plt.title("GAUSSIANO 61X61")
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(img_filtro_mediana2)
plt.title("MEDIANA 61X61")
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.imshow(img_filtro_bilateral2)
plt.title("BILATERAL 61X61")
plt.xticks([])
plt.yticks([])

plt.show()

# Filtros de borda

#Laplacian
img_1 = cv2.Laplacian(img_rgb,cv2.CV_64F) #Função para aplicação do filtro em imagem 64b.
abs_164f = np.absolute(img_1)             #Transormando em numeros inteiros.
img_1 = np.uint8(abs_164f)                #Transformando para inteiro de 8b.

#Sobel
img_sx = cv2.Sobel(img_rgb,cv2.CV_64F,0,1,ksize=5)  #Função para aplicação do filtro (imagem, ident.64b, eixoy,eixo x e kernel).
abs_sx64f = np.absolute(img_sx)
img_sx = np.uint8(abs_sx64f)

img_sy = cv2.Sobel(img_rgb,cv2.CV_64F,1,0,ksize=5)
abs_sy64f = np.absolute(img_sy)
img_sy = np.uint8(abs_sy64f)

#Bordas Canny
edges = cv2.Canny(img_rgb, 100,200)                 #Função (imagem rgb, minimo e maximo do gradiente de borda (delimitação da borda). Os valores intermediários também são processados pelo algoritmo por conectividade com as de valores acima de 200.

#Apresentação das imagens
plt.figure('Imagens- Filtros de Bordas')
plt.subplot(231)
plt.imshow(img_rgb)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(img_1)
plt.title("Laplacian")
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(img_sx)
plt.title("Sobel eixo X")
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(img_sy)
plt.title("Sobel eixo Y")
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.imshow(edges)
plt.title("Canny")
plt.xticks([])
plt.yticks([])

plt.show()

print('-='*50)
print('')
#######################################################################################################
################### ESCALHO DO CANAL PARA SEGUIMENTAÇÃO################################################
#######################################################################################################
img_Lab = cv2.cvtColor(img_filtro_mediana1,cv2.COLOR_BGR2Lab)
img_HSV = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
img_YCrCb = cv2.cvtColor(img_filtro_mediana1,cv2.COLOR_BGR2YCR_CB)

#### RGB ###
hist_r = cv2.calcHist([img_filtro_mediana1],[0], None, [256],[0,256]) #[0] queremos acessa o canal 0.(RED)
hist_g = cv2.calcHist([img_filtro_mediana1],[1], None, [256],[0,256]) #[1] queremos acessa o canal 1.(GREEN)
hist_b = cv2.calcHist([img_filtro_mediana1],[2], None, [256],[0,256]) #[2] queremos acessa o canal 2.(BLUE)


plt.figure('RGB')
plt.subplot(2,3,1)
plt.imshow(img_filtro_mediana1[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_filtro_mediana1[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_filtro_mediana1[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_r,color = 'r')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_r,color = 'g')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_r,color = 'b')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')


############### Lab ###################
hist_L = cv2.calcHist([img_Lab],[0],None,[256],[0,256])
hist_a = cv2.calcHist([img_Lab],[1],None,[256],[0,256])
hist_b = cv2.calcHist([img_Lab],[2],None,[256],[0,256])

plt.figure('Lab')
plt.subplot(2,3,1)
plt.imshow(img_Lab[:,:,0],cmap = 'gray')
plt.title('Segmentada - L')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_Lab[:,:,1],cmap = 'gray')
plt.title('Segmentada - a')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_Lab[:,:,2],cmap = 'gray')
plt.title('Segmentada - b')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_L, color = 'black')
plt.title('Histograma - L')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_a, color = 'black')
plt.title('Histograma - a')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_b, color = 'black')
plt.title('Histograma - b')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

########### HSV ##########
hist_H = cv2.calcHist([img_HSV],[0],None,[256],[0,256])
hist_S = cv2.calcHist([img_HSV],[1],None,[256],[0,256])
hist_V = cv2.calcHist([img_HSV],[2],None,[256],[0,256])

plt.figure('HSV')
plt.subplot(2,3,1)
plt.imshow(img_HSV[:,:,0],cmap = 'gray')
plt.title('Segmentada - H')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_HSV[:,:,1],cmap = 'gray')
plt.title('Segmentada - S')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_HSV[:,:,2],cmap = 'gray')
plt.title('Segmentada - V')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_H, color = 'black')
plt.title('Histograma - H')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_S, color = 'black')
plt.title('Histograma - S')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_V, color = 'black')
plt.title('Histograma - V')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')



######## YCrCb ##########
hist_Y = cv2.calcHist([img_YCrCb],[0],None,[256],[0,256])
hist_CR = cv2.calcHist([img_YCrCb],[1],None,[256],[0,256])
hist_CB = cv2.calcHist([img_YCrCb],[2],None,[256],[0,256])

plt.figure('YCrCb')
plt.subplot(2,3,1)
plt.imshow(img_YCrCb[:,:,0],cmap = 'gray')
plt.title('Segmentada - Y')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_YCrCb[:,:,1],cmap = 'gray')
plt.title('Segmentada - Cr')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_YCrCb[:,:,2],cmap = 'gray')
plt.title('Segmentada - Cb')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.plot(hist_Y, color = 'black')
plt.title('Histograma - Y')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,5)
plt.plot(hist_CR, color = 'black')
plt.title('Histograma - Cr')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')

plt.subplot(2,3,6)
plt.plot(hist_CB, color = 'black')
plt.title('Histograma - Cb')
plt.xlim([0,256])
plt.xlabel('Valores de Pixels')
plt.ylabel('Número de Pixels')
plt.show()
#####################################################################################################################



print('c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o '
      'reconhecimento de contornos, identifique e salve os objetos de interesse. Além disso, acesse'
      ' as bibliotecas Opencv e Scikit-Image, verifique as variáveis que podem ser mensuradas e extraia '
      'as informações pertinentes (crie e salve uma tabela com estes dados). '
      'Apresente todas as imagens obtidas ao longo deste processo.')



#L,a,b = cv2.split(img_Lab)
H,S,V = cv2.split(img_HSV)

s_f = cv2.medianBlur(S,5)
(L, img_limiar) = cv2.threshold(s_f,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_segmentada1 = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar)

plt.figure('seg')
plt.subplot(1,3,1)
plt.title('Imagem original')
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.title('Imagem com limiar de OTSU')
plt.imshow(img_limiar, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.title('Imagen segmentada')
plt.imshow(img_segmentada1)
plt.xticks([])
plt.yticks([])
plt.show()
################## RESPOSTAS NO CONSOLE########################
# identificar os contornos da imagem

mascara = img_limiar.copy() #para não modicar a imagen original
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar[y:y+h,x:x+w]
	obj_rgb = img_segmentada1[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('f'+str(i+1)+'.png',obj_bgr)
	cv2.imwrite('fbin'+str(i+1)+'.png',obj)

	regiao = regionprops(obj) #https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
	print('Folha: ', str(i+1)) #folha 1
	print('Dimensão da Imagem: ', np.shape(obj)) #tamnho do retangula
	print('Medidas Físicas')
	print('Centroide: ', regiao[0].centroid) # acessar o centroide, ou seja onde está a posição central das folhas[0] posição 0
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length) #razão do eixo maiior com menor
	area = cv2.contourArea(c) # foi feita a opção por utillizar o open cv, mas poderia ser o skinage tambem, mas quando for determinar o objeto é obj (c é o contorno)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True)) # no caso do perimetro também é a mesma coisa, usamos o c de contorno, se fosse skimage seria obj

##obter mediada de cor, cv2.minMaxLoc = > podemos obter o maximo e minimo dentro da matriz que estamos trabalhando, onbtemos a posição
## com essa função são retornados os valore de max e min e os locais deles.
## nesta função devemos colocar o obj ou matriz RGB e uma mascara binaria(região branca da imagem grao ou folha)
## podemos obter a média dos pixel de canda canal RGB

	print('Medidas de Cor')
	min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_r, ' - Posição: ', min_loc_r)
	print('Valor Máximo no R: ', max_val_r, ' - Posição: ', max_loc_r)
	med_val_r = cv2.mean(obj_rgb[:,:,0], mask=obj)
	print('Média no Vermelho: ', med_val_r)

	min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_g, ' - Posição: ', min_loc_g)
	print('Valor Máximo no G: ', max_val_g, ' - Posição: ', max_loc_g)
	med_val_g = cv2.mean(obj_rgb[:,:,1], mask=obj)
	print('Média no Verde: ', med_val_g)

	min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no B: ', min_val_b, ' - Posição: ', min_loc_b)
	print('Valor Máximo no B: ', max_val_b, ' - Posição: ', max_loc_b)
	med_val_b = cv2.mean(obj_rgb[:,:,2], mask=obj)
	print('Média no Azul: ', med_val_b)
	print('-'*50)



########################################################################################################################
print('Total de folhas: ', len(cnts))
#fazer a contagem do numero de folhas/graos, contabiliza o numero de contornos
print('-'*50)

seg = img_segmentada1.copy()
cv2.drawContours(seg,cnts,-1,(255,0,0),8)
# essa função cria um contorno em cada uma dos graos ou folhas
# seg - é a imagem
# cnts contorno
# -1 indica que é para fazer em todos os grãos
# (0,255,0) exemplo do professor(verde) (255,0,0) isso é uma tucla que informa a cor da linha do contorno (vermelho)
# 2 representa a espesura da linha que quermos colocar na imagem.

plt.figure('Folhas')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('folhas')

plt.subplot(1,2,2)
plt.imshow(obj_rgb)
plt.xticks([])
plt.yticks([])
plt.title('folha')
plt.show()

##########################################SALVAR EM UMA TABELA CSV ###################################################

# identificar os contornos da imagem
mascara = img_limiar.copy() #para não modicar a imagen original
mask = np.zeros(img_rgb.shape,dtype = np.uint8)
cnts = cv2.findContours(mascara,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

dimen = []
for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar[y:y+h,x:x+w]
	obj_rgb = img_segmentada1[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('folha'+str(i+1)+'.png',obj_bgr)


	area = cv2.contourArea(c)
	razao = round((h/w), 2)
	perim = round(cv2.arcLength(c, True), 2)
	tam_ret = np.shape(obj)

	regiao = regionprops(obj)
	rm = round(regiao[0].minor_axis_length, 2)
	rmai = round(regiao[0].major_axis_length, 2)
	cen = regiao[0].centroid
	dimen += [[str(i + 1), str(h), str(w), str(area), str(razao),
			   str(perim), str(tam_ret), str(rm), str(rmai), str(cen)]]

dados_folhas = pd.DataFrame(dimen)
dados_folhas = dados_folhas.rename(columns={0: 'FOLHA', 1: 'ALTURA', 2: 'LARGURA', 3: 'AREA', 4: 'RAZAO',
											5: 'PERIMETRO', 6: 'TAMANHO RET', 7:'EIXO MENOR', 8:'EIXO MAIOR',
											9:'CENTROIDE'})
dados_folhas.to_csv('medidas.csv', index=False)

print('d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.')

img_segmentada_backup = img_segmentada1.copy()
img_segmentada = cv2.cvtColor(img_segmentada1,cv2.COLOR_RGB2HSV)

plt.figure('Objetos')
for (i, c) in enumerate(cnts):
	(x,y,w,h) = cv2.boundingRect(c)
	print('Objeto # %d' % (i+1))
	print(cv2.contourArea(c))
	obj = img_limiar[y:y+h,x:x+w]
	obj_rgb = img_segmentada[y:y+h,x:x+w]

	grafico = True
	if grafico == True:

		hist_segmentada_r = cv2.calcHist([obj_rgb], [0], obj, [256], [0, 256])
		hist_segmentada_g = cv2.calcHist([obj_rgb], [1], obj, [256], [0, 256])
		hist_segmentada_b = cv2.calcHist([obj_rgb], [2], obj, [256], [0, 256])
		# obj = img_rgb[y:y + h, x:x + w]


		plt.subplot(3,3,2)
		plt.imshow(obj_rgb)
		plt.title('Objeto: ' + str(i+1))

		plt.subplot(3, 3, 4)
		plt.imshow(obj_rgb[:,:,0],cmap='gray')
		plt.title('Objeto: ' + str(i + 1))

		plt.subplot(3, 3, 5)
		plt.imshow(obj_rgb[:,:,1],cmap='gray')
		plt.title('Objeto: ' + str(i + 1))

		plt.subplot(3, 3, 6)
		plt.imshow(obj_rgb[:,:,2],cmap='gray')
		plt.title('Objeto: ' + str(i + 1))

		plt.subplot(3, 3, 7)
		plt.plot(hist_segmentada_r, color='r')
		plt.title("Histograma - R")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")

		plt.subplot(3, 3, 8)
		plt.plot(hist_segmentada_g, color='g')
		plt.title("Histograma - G")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")

		plt.subplot(3, 3, 9)
		plt.plot(hist_segmentada_b, color='b')
		plt.title("Histograma - B")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")
		plt.show()


	else:
		pass



print('e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as '
	  'imagens obtidas neste processo.')
print('-'*55)
print('INFORMAÇÕES DA IMAGEM')
print('-'*55)
print('Dimensão: ',np.shape(img_rgb))
#primeiramente vamos calcular a dimenção desta imagem, ou seja, número de pixels.
# cada pixel será considerado como uma Observação, sendo RGB
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])
print('-'*55)


# Formatação da imagem para uma matriz de dados
pixel_values = img_rgb.reshape((-1, 3))             # muda o formato desta imagem -1 referente a linhas( cada pixel se torna uma linha)
                                                # 3 refere-se as colunas cada canal (RGB)
# Conversão para Decimal
pixel_values = np.float32(pixel_values)         # trabalha com numeros decimais,
print('-'*55)
print('Dimensão Matriz: ',pixel_values.shape)   # dimensão da matriz, temos 3 colunas e x pixels de linha
print('-'*55)
########################################################################################################################
# K-means
# Critério de Parada (mediada de precisão 0,2 ou numero max de iterações sendo 100,)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Número de Grupos (k)
# nos podemos determminar o numero de grupos neste caso foi escolhido a maça G1, o fundo G2 e As demias maças G3.
k = 2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# pixel_values => é uma matriz , pois estamos trabalhando com uma tecnica de agrupamento.
# k => é o numero de grupos
# None => quer dizer que os rotolos dos grupos não serão especificados, estes serão numeros (1,2e3).
# criteria => foi nostrada acima
# 10 => iniciação dos centroides co K-means, rodando esta 10 vezes.
# cv2.KMEANS_RANDOM_CENTERS => estipular os centroides no início.

### RESPOSTAS ###
# dist => Soma Quadrados das Distâncias de Cada Ponto ao Centroide, é uma mendiada dada em pixels
# label => são as informacoes 0, 1 e 2

print('-'*80)
print('Soma Quadrados das Distâncias de Cada Ponto ao Centro: ', dist)
print('-'*80)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels)) #valores unicos de cada grupo que determinamos
print('Tipo labels: ', type(labels))
# flatten the labels array
labels = labels.flatten() # transformação em um vetor como se fosse uma lista IMPORTANTE PARA OS PROXIMOS PASSOS
print('-'*80)
print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))
print('-'*80)

# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)     #RETORNAR A CONTAGEM DE CADA GRUPO(012)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))         # 1 POIS QUEMOS ESTE NA
contagens = np.reshape(contagens,(len(contagens),1))            # FAZ UMA CONTAGEM
hist = np.concatenate((val_unicos,contagens),axis=1)            # FAZ UMA CONCATENAÇÃO
print('Histograma')
# MOSTRA A QUANTIDADE DE PIXEL QUE TEM EM CADA GRUPO.
print(hist)
print('-'*80)

print('Centroides Decimais')
# CADA LINHA É UM GRUPO E TEM INFORMAÇÃO DE CADA CANAL (RGB), COM VALORES DECIMAIS
print(centers)
print('-'*80)
# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)
# CADA LINHA É UM GRUPO E TEM INFORMAÇÃO DE CADA CANAL (RGB), COM VALORES INTERIRO DE 8 BITS
print('-'*80)
print('Centroides uint8')
print(centers)
print('-'*80)
########################################################################################################################
# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels] # CRIAR UMA MATRIZ
print('-'*80)
print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:]) # PEGA DA LINHA 0 ATÉ A LINHA 5. APENAS UM PEDACINHO.
print('-'*80)
########################################################################################################################
# Reformatar a matriz na imagem de formato original
# PEGAMOS A MATRIZ QUE CADA LINHA LINHA ERA UM PIXEL E CADA COLUNA ERA UM CANAL RGB E AGORA VAMOS VOLTAR ESTA PARA IMGEM COM DIMENSAO DA IMAGEM
img_segmentada = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb) # CRIAR UMA NOVA IMAGEM
matriz_or_01 = original_01.reshape((-1, 3)) # VAMOS PEGAR UMA IMAGEM E TRANSFORMA EM uma matriz em que as  LINHAs são os (PIXEL) E as COLUNAs (RGB)
matriz_or_01[labels != 0] = [0, 0, 0] # label => sao os rotolos que temos -> A posição (!= 0) será zerada ([0,0,0,)
img_final_01 = matriz_or_01.reshape(img_rgb.shape) # volta para imagem, mas somente com as informaçoes de cores da regiao (!=0)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels != 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)

# Grupo 3
original_03 = np.copy(img_rgb)
matriz_or_03 = original_03.reshape((-1, 3))
matriz_or_03[labels != 2] = [0, 0, 0]
img_final_03 = matriz_or_03.reshape(img_rgb.shape)

########################################################################################################################
# Apresentar Imagem
plt.figure('Imagens')
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('ORIGINAL')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(img_segmentada)
plt.title('ROTULOS')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2 - Segmentação das folhas')
plt.xticks([])
plt.yticks([])



plt.show()



print('f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as '
	  'imagens obtidas neste processo. ')
'''

Em portugues esta é chamada de bacia hidrografica

The watershed algorithm is a classic algorithm used for segmentation and is especially useful when extracting touching
 or overlapping objects in images , such as the coins in the figure above.

Using traditional image processing methods such as thresholding and contour detection, 
we would be unable to extract each individual coin from the image — but by leveraging the watershed algorithm, we are able to detect and extract each coin without a problem.

When utilizing the watershed algorithm we must start with user-defined markers. 
These markers can be either manually defined via point-and-click, or we can automatically or 
heuristically define them using methods such as thresholding and/or morphological operations.
'''


img_HSV = cv2.cvtColor(img_filtro_mediana1,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_HSV)

limiar, mascara = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_dist = ndimage.distance_transform_edt(mascara)
# ndimage.distance_transform_edt: Calcula a distância euclidiana até o zero
# mais próximo (ou seja, pixel de fundo)para cada um dos pixels de primeiro plano.

# na verdade calcula a distancia do fundo até a borda que será cheia.

max_local = peak_local_max(img_dist, indices=False, min_distance=100, labels=mascara)

# peak_local_max: retorna uma matriz booleana com os picos da imagem (maior valor que temos dentro da matriz = parte vermelha max)
# indices false = marcamos falso pois não queremos o local, e sim uma imagem com valor verdadeiro
# baseados nas distâncias
# min_distance: Número mínimo de pixels que separam os picos (meio no chute, depende do tamanho da imagem)
# labels=mascara (pois queremos olhar apenas a mascara, que foi limearizada)

print('-'*50)
print('Número de Picos')
print(np.unique(max_local,return_counts=True))
# mostra na saida false o fundo, true os picos das imagens(folhas, batata o que for)

print('-'*50)
marcadores,n_marcadores = ndimage.label(max_local, structure=np.ones((3, 3)))
# # structure=np.ones((3, 3)) serve para ir no valor do pico, e nos seus visinho neste caso 8 vizinhos, analise de conectividade


# Realiza marcação dos picos
print('Análise de conectividade - Marcadores')
print(np.unique(marcadores,return_counts=True))
#faz a marcação de numero de picos
print('-'*50)

img_ws = watershed(-img_dist, marcadores, mask=mascara)
# -img_dist = > imagem das distancias (coloca negativo para inveter os picos por vales)
# marcadores é a regiao do fundo, onde vamos começar a inundar.
# mask

# não vamos inundar até a borda,  pois o fundo(meio da folha é mais fundo ) quando chegar na borda paramos de encher
print('Imagem Segmentada - Watershed')
print(np.unique(img_ws,return_counts=True)) # a imagem (img_ws é a imagem marcada com rotulos)
# o comando acima no mostra o pixels associado com a posição.
print("Número de Folhas: ", len(np.unique(img_ws)) - 1) #nos dá a quantidade de folhas
# temos que colocar -1 para deixar o fundo da imagem
print('-'*50)

img_final = np.copy(img_rgb) # criar uma copia
img_final[img_ws != 2] = [0,0,0] # Acessando a folha 2. Ao determinar a folha que vamos acessar (!=2) e
# posteriormente zerando os demais pontos ([0,0,0]) temos a seleção. dessa maneira todos os valore vão fica preto e a folha não.
########################################################################################################################
plt.figure('Watershed')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('ORIGINAL')

plt.subplot(2,3,2)
plt.imshow(s,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('S')

plt.subplot(2,3,3)
plt.imshow(mascara,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Mascara')

plt.subplot(2,3,4)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Distância')

plt.subplot(2,3,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Folhas')

plt.subplot(2,3,6)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('SELEÇÃO')

plt.show()


print('g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed)'
	  ' e identifique as potencialidades de cada delas. ')
plt.figure('Imagens')
plt.subplot(1,3,1)
plt.imshow(img_ws)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada Wathershed')

plt.subplot(1,3,2)
plt.imshow(img_segmentada1)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada OTSU')

plt.subplot(1,3,3)
plt.imshow(img_final_02)
plt.title('Segmentada K-MEANS')
plt.xticks([])
plt.yticks([])

plt.show()

