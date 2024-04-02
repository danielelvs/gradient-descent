import os
import numpy as np
from sklearn.datasets import fetch_openml # para baixar o dataset do scikit-learn
from sklearn.model_selection import train_test_split # para dividir o dataset para ser treinado
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Imagem é altura x largura x 3 (rgb) = isso é o tamanho da imagem
# se tivermos um conjunto de dados de imagens onde cada imagem tem dimensões altura×largura×3 (considerando as três bandas de cores RGB), então a matriz terá dimensões m×n, onde:
# - m é o número de exemplos de treinamento (ou o número total de imagens no conjunto de dados);
# - n é o número total de características (ou o número total de pixels em cada imagem).
# Y é um vetor com os rótulos. Deve ter a mesma quantidade de linhas ou colunas que X
# ele deve possuir por exemplo um vetor de inteiros onde cada número representa a classe de uma imagem (ex.: 0 para gatos, 1 para cachorros, 2 para pássaros).

class Gradient:

  def __init__(self, showPlot = False):
    self.plot = showPlot

  def setDatasetName(self, name):
    self.datasetName = name

  def setDatasetDir(self, dir):
    self.datasetDir = f'{dir}/{self.datasetName}'

  def setXy(self, dataset):
    self.X, self.y = dataset

  def setDadosTreinamento(self, dadosTreinamento):
    self.X_treino, self.X_teste, self.y_treino, self.y_teste = dadosTreinamento

  def cria_diretorios(self):
    """
      Cria diretório inicial que será utilizado para inserir o dataset
    """
    assert self.datasetName is not None, "O nome do dataset não foi definido"
    assert self.datasetDir is not None, "O diretório do dataset não foi definido"

    # cria o diretório, se não existir um dos caminhos intermediários, o cria.
    # Quando o diretório já existir, não emite mensagem de erro (exist_ok)
    os.makedirs(name=self.datasetDir, exist_ok=True)

  def get_dataset(self):
    """
      Cria diretórios de X (pixels das imagens) e y (rótulos das mesmas).

      Valida se os mesmos diretórios existem:
          Se não, é feito o download do dataset e atribuído as respectivas variáveis de X e y. Após, o NumPy as salva nos diretórios já vetorizando os dados.

          Se sim, o NumPy os carrega nas variáveis respectivas de X e y;
    """

    caminho_X = f'{self.datasetDir}/X.npy' # imagens
    caminho_y = f'{self.datasetDir}/y.npy' # rótulos/labels

    # se não existir esses diretórios
    if not (os.path.exists(caminho_X) and os.path.exists(caminho_y)):
      # Baixa o dataset
      dataset_mnist = fetch_openml(name=self.datasetName, as_frame=False, cache=True, return_X_y=True)

      # O dataset é uma tupla com 2 valores.
      # Atribui as imagens para X e os labels para y.
      # X = ndarray with shape (70000, 784)
      # y = ndarray with shape (70000,)
      self.setXy(dataset_mnist)
      # self.X, self.y = dataset_mnist

      # Criando a matriz.
      # Salva o dataset como um array numpy
      np.save(caminho_X, self.X.astype(np.float32))
      np.save(caminho_y, self.y)
      print(f'Dataset MNIST baixado e salvo em {self.datasetDir}.\n\n')

    else: # se já existir, carrega.
      self.X = np.load(caminho_X, allow_pickle=True)
      self.y = np.load(caminho_y, allow_pickle=True)
      print(f'Dataset MNIST já existe em {self.datasetDir}.\n\n')

    # mostra a quantidade de elementos em X, min() e max() [?]
    # o que seria X.shape, X.min() e X.max()
    print(f'X.shape: {self.X.shape}\nX.min(): {self.X.min()}\nX.max(): {self.X.max()}\n\n')

  def get_plot(self):
    """
      Exibe exemplos do dataset, mostrando alguns exemplos das imagens treinadas por cada classe (rótulo).

      Aqui é exibido 10 classes por 10 exemplos/variações
    """

    classes = [c for c in range(10)] # loop inline. Atribui c a classes seguido do for sobre o tamanho 10
    qtde_colunas = len(classes)
    qtde_linhas = 10

    for c in classes:
      # retornará os rótulos de y iguais a classe corrente em um array
      indices = np.flatnonzero(self.y == str(c))

      # sobrescreve a variável, selecionamento randomicamente os itens
      # na coluna 0 (c=0) gerou = array([37153, 12554, 46560, 11605, 1916])
      indices = np.random.choice(a=indices, size=qtde_linhas, replace=False)

      for i, indice in enumerate(indices):
        # valores aleatórios, para pegar uma imagem aleatória?
        # na coluna 0 (c=0) gerou em cada loop = 0, 11, 21, 31, 41
        plot_indice = i * qtde_colunas + c + 1

        plt.subplot(qtde_linhas, qtde_colunas, plot_indice) # monta a célula (linha X coluna).
        plt.imshow(X=self.X[indice].reshape((28, -1)).astype('uint8'), cmap='bone') # insere a imagem na linha da coluna (classe) da subplot. O que seria o 28, -1??
        plt.axis('off')

        if i == 0: # coloca a label da classe, sempre na primeira vez ao montar a coluna
          plt.title(c)

    plt.show()

  def divide_dataset(self):
    """
      Divide o dataset para realizar o treinamento e teste, sem validação
    """

    np.random.seed(10) # quantidade de vezes a ser reproduzido. Seria épocas?
    self.y = self.y.astype(int) # converte os rótulos de string para int

    # espera-se os arrays de treino e a porcentagem que será repartida o dataset para treino
    # o retorno será atribuído para cada variável respectiva.
    self.setDadosTreinamento(train_test_split(self.X, self.y, test_size=0.25))

    print(f'Tamanho para treino:\n X: {len(self.X_treino)}\n y: {len(self.y_treino)}\n\n')
    print(f'Tamanho para teste:\n X: {len(self.X_teste)}\n y: {len(self.y_teste)}\n\n')

  def svm_loss(self, W):
    """
      Calcula a LOSS e Gradient para a classificação linear SVM
        X (numpy.ndarray): Dados de entrada (qtd_treinamento, num_features). self.X_treino
        y (numpy.ndarray): Matriz dos rótulos (qtd_treinamento). self.y_treino

      Parâmetros:
        W (numpy.ndarray): Matriz de peso (num_features, num_classes).

      Retorna:
        float: O valor da loss.
    """

    scores = self.X_treino.dot(W) # obtém as pontuações
    qtd_treinamento = self.X_treino.shape[0] # Obtém a quantidade de exemplos de treinamento
    scores_correta_classes = scores[np.arange(qtd_treinamento), self.y_treino] # obtém as pontuações corretas das classes
    margens = np.maximum(0, scores - scores_correta_classes[:, np.newaxis] + 1) # cálcula as margens
    margens[np.arange(qtd_treinamento), self.y_treino] = 0 # não é considerada a margem para a classe correta
    loss = np.sum(margens) / qtd_treinamento # cálcula a loss

    return loss

  def previsoes(self, W):
    scores = self.X_teste.dot(W)
    classes_previstas = np.argmax(scores, axis=1)
    return classes_previstas

  def gera_plot_classificacao(self, yPrevisao):
    labels = [classes for classes in range(10)]
    calculo_relatorio = confusion_matrix(self.y_teste, yPrevisao)
    dados = confusion_matrix(self.y_teste, yPrevisao, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=dados, display_labels=labels)
    display.plot()

    relatorio = classification_report(self.y_teste, yPrevisao, target_names=[str(label) for label in range(10)])
    print(f'\nRelatório da Classificação: \n {relatorio}')

    plt.show()

  def gera_plot_reshape(self, melhorW):
    W_reshaped = melhorW.reshape(28, 28, -1)
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))

    for i in range(10):
      valor_ax = ax[i // 5, i % 5]
      valor_ax.imshow(W_reshaped[:, :, 1], cmap='bone')
      valor_ax.axis("off")
      valor_ax.set_title(f"{i}")

    plt.show()







  def treinamento(self):
    qtde_interacoes = 30 # épocas?
    qtde_classes = 10
    qtde_caracteristicas = self.X_treino.shape[1]

    melhor_loss = float('inf')
    melhor_W = None # peso

    historico_loss = []
    historico_melhor_loss = []

    for interacao in range(qtde_interacoes):
      W = np.random.randn(qtde_caracteristicas, qtde_classes) * 0.0001 # inicialização dos pesos aleatório
      loss = self.svm_loss(W)
      historico_loss.append(loss)

      if loss < melhor_loss:
        melhor_loss = loss
        melhor_W = W

      historico_melhor_loss.append(melhor_loss)
      print(f'Interação {1 + interacao}:\n Loss: {loss}.\n Melhor Loss: {melhor_loss}.\n')

    print(f'Melhor loss encontrada: {melhor_loss}.\n')

    #####

    # plt.figure(figsize=(10, 6))
    # plt.plot(historico_loss, label='Loss')
    # plt.plot(historico_melhor_loss, 'r--', label="Melhor Loss")
    # plt.xlabel('Interações')
    # plt.ylabel('Loss')
    # plt.title('Loss vs. Interações')
    # plt.legend()
    # plt.show()

    #####

    previsao_y = self.previsoes(melhor_W)
    print(f"Previsão vs. Rótulos corretas: \n {previsao_y} \n {self.y_teste}")

    #####

    # self.gera_plot_classificacao(previsao_y)

    #####

    self.gera_plot_reshape(melhor_W)




def main():
  g = Gradient()

  g.setDatasetName('mnist_784')
  g.setDatasetDir('dataset')

  g.cria_diretorios()
  g.get_dataset()

  if (g.plot == True):
    g.get_plot()

  g.divide_dataset()
  g.treinamento()

if __name__ == "__main__":
  main()

# https://colab.research.google.com/drive/18HfsjtihT85Pb7xAdJ5rGE2ml9KyuVul?authuser=1#scrollTo=OKpPaEsBooVJ
