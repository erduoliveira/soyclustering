from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import metrics


class KmeansTest:

    def __init__(self):
        self.numero_clusters = 10
        self.figure = 0
        self.true_label = np.array(self.get_true_label()) 

    def main(self):
        #self.analise('BIC')
        #self.analise('FCTH')
        self.analise('Gabor')
        #self.analise('Tamura')
        plt.show()

    def analise(self, descritor):
        self.get_samples(descritor)
        modelo = KMeans(n_clusters=self.numero_clusters).fit(self.amostras)
        pred_label = modelo.labels_
        print(descritor + ' Scores')
        print('Fowlkes-Mallows score: ' + str(metrics.fowlkes_mallows_score(self.true_label, pred_label)))
        print('Davies-Bouldin score: ' + str(metrics.davies_bouldin_score(self.amostras, pred_label)))
        print(
            'Calinski and Harabasz score: ' + str(metrics.calinski_harabasz_score(self.amostras, pred_label)) + '\n\n')
        #self.visualiza_clusterizacao(descritor, pred_label)

    def visualiza_clusterizacao(self, descritor, pred_label):
        tsne = TSNE()
        visualizacao = tsne.fit_transform(self.amostras)
        self.nova_figura()
        plt.suptitle(descritor + ' - Kmeans')
        plt.subplot(1, 2, 1)
        ax = sns.scatterplot(x=visualizacao[:, 0],
                             y=visualizacao[:, 1],
                             hue=self.true_label,
                             palette=sns.color_palette('Set1', self.numero_clusters)
                             )
        ax.get_legend().remove()
        ax.set_title('True label')
        plt.subplot(1, 2, 2)
        ax = sns.scatterplot(x=visualizacao[:, 0],
                             y=visualizacao[:, 1],
                             hue=pred_label,
                             palette=sns.color_palette('Set1', self.numero_clusters)
                             )
        ax.get_legend().remove()
        ax.set_title('Pred label')

    def get_samples(self, descritor):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/' + descritor + '.txt')]
        aux = []
        for linha in linhas[1:]:  # Skip primeira linha (Contem apenas metadata)
            auxList = linha.split(' ')
            auxList = list(filter(None, auxList))  # Remove espacos em branco
            auxList.pop(0)  # Id da amostra no arquivo
            auxList.pop(0)  # Especie de tentativa de especificar a classe
            auxList = list(map(float, auxList))
            auxList = np.array(auxList)
            aux.append(auxList)
        self.amostras = np.array(aux)

    def nova_figura(self):
        plt.figure(self.figure)
        self.figure = self.figure + 1

    def get_true_label(self):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/true_label.txt')]
        true_label = []
        for linha in linhas:
            true_label.append(linha)
        return true_label


if __name__ == '__main__':
    KmeansTest().main()
