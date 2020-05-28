from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import metrics


class NumeroClusters:

    def __init__(self):
        self.figure = 0
        self.cluster_max = 20

    def main(self):
        self.get_samples('BIC')
        self.get_elbow('BIC')
        self.get_silhouette('BIC')
        plt.show()

    def get_silhouette(self, descritor):
        self.nova_figura()
        silhouette = [self.silhouette_kmeans(numero_clusters) for numero_clusters in range(2, self.cluster_max)]
        silhouette = pd.DataFrame(silhouette, columns=['grupos', 'score'])
        plt.plot(silhouette.grupos, silhouette.score)
        plt.suptitle('Silhouette method - ' + descritor)
        plt.xlabel('Numero de clusters')
        plt.ylabel('Coeficiente de silhueta')
        plt.xticks(silhouette.grupos)
        #plt.ylim(min(silhouette.score), silhouette.score[1])
        #plt.xlim(2, 50)

    def silhouette_kmeans(self, numero_clusters):
        modelo = KMeans(n_clusters=numero_clusters).fit(self.amostras)
        return [numero_clusters, metrics.silhouette_score(self.amostras, modelo.labels_)]

    def elbow_kmeans(self, numero_clusters):
        modelo = KMeans(n_clusters=numero_clusters).fit(self.amostras)
        return [numero_clusters, modelo.inertia_]

    def get_elbow(self, descritor):
        self.nova_figura()
        elbow = [self.elbow_kmeans(numero_clusters) for numero_clusters in range(1, self.cluster_max)]
        elbow = pd.DataFrame(elbow, columns=['grupos', 'inertia'])
        plt.plot(elbow.grupos, elbow.inertia)
        plt.suptitle('Elbow method - ' + descritor)
        plt.xlabel('Numero de clusters')
        plt.ylabel('SSE')
        plt.xticks(elbow.grupos)
        plt.ylim(min(elbow.inertia), elbow.inertia[1])
        plt.xlim(2, self.cluster_max)

    def get_samples(self, descritor):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas-d6/' + descritor + '.txt')]
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


if __name__ == '__main__':
    NumeroClusters().main()
