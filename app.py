import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Função para calcular as propriedades da imagem
def calcular_propriedades_imagem(imagem):
    altura, largura, canais = imagem.shape
    media_cores = np.mean(imagem, axis=(0, 1))
    desvio_padrao_cores = np.std(imagem, axis=(0, 1))
    imagem_lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2Lab)
    canal_l = imagem_lab[:, :, 0]
    cores_unicas = np.unique(canal_l)
    return {
        'Altura': altura,
        'Largura': largura,
        'Média das Cores': media_cores,
        'Desvio Padrão das Cores': desvio_padrao_cores,
        'Quantidade de Cores': len(cores_unicas)
    }

# Função para aplicar o algoritmo K-Médias na imagem
def aplicar_kmeans(imagem, k):
    pixels = imagem.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centros = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)
    imagem_segmentada = centros[labels.flatten()].reshape(imagem.shape)
    return imagem_segmentada


# Diretório onde as imagens estão localizadas
diretorio_imagens = '/content/images/'

diretorio_saida = '/content/imagens_segmentadas/'


# Lista de nomes de arquivos de imagem no diretório
nomes_imagens = os.listdir(diretorio_imagens)
# Loop através das imagens no diretório
for nome_imagem in nomes_imagens:
    # Carregar a imagem de entrada
    caminho_imagem = os.path.join(diretorio_imagens, nome_imagem)
    imagem_original = cv2.imread(caminho_imagem)

    # Calcular propriedades da imagem original
    propriedades_original = calcular_propriedades_imagem(imagem_original)
    tamanho_do_arquivo_original = os.path.getsize(caminho_imagem)
    tamanho_original =  f'{tamanho_do_arquivo_original / 1024:.2f} KB'
    print(f'\n\n\nTamanho da imagem Original = {tamanho_original}')

    # Aplicar o algoritmo K-Médias para diferentes valores de k
    valores_de_k = [7,14,21,28,35,42,49]
    resultados_kmeans = []

    for k in valores_de_k:
        imagem_segmentada = aplicar_kmeans(imagem_original, k)
        propriedades_segmentada = calcular_propriedades_imagem(imagem_segmentada)
        resultados_kmeans.append((k, imagem_segmentada, propriedades_segmentada))


    # Exibir resultados
    print(f'Propriedades da imagem original:')
    for chave, valor in propriedades_original.items():
      print(f'{chave}: {valor}')

    # Printa a imagem original
    cv2_imshow(imagem_original)

    print(f'Resultados para a imagem: {nome_imagem}')
    for k, imagem_segmentada, propriedades_segmentada in resultados_kmeans:
        print(f'\nResultados para k = {k}:')


        # Salvar a imagem segmentada
        nome_saida = f'{os.path.splitext(nome_imagem)[0]}_k{k}.png'
        caminho_saida = os.path.join(diretorio_saida, nome_saida)
        cv2.imwrite(caminho_saida, imagem_segmentada)
        tamanho_do_arquivo = os.path.getsize(caminho_saida)
        propriedades_segmentada['Tamanho da Imagem Segmentada'] = f'{tamanho_do_arquivo / 1024:.2f} KB'

        print(f'Propriedades da imagem segmentada:')
        for chave, valor in propriedades_segmentada.items():
            print(f'{chave}: {valor}')


        cv2_imshow(imagem_segmentada)

    # Plotar as imagens
    for k, imagem_segmentada, _ in resultados_kmeans:
        plt.subplot(7, 49, k)
        plt.imshow(cv2.cvtColor(imagem_segmentada, cv2.COLOR_BGR2RGB))
        plt.title(f'k = {k}')

    plt.show()
