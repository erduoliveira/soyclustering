from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import metrics

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from pyclustering.cluster.agglomerative import agglomerative
from pyclustering.cluster.birch import birch
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.optics import optics
from pyclustering.cluster.rock import rock

from pyclustering.cluster import cluster_visualizer


class Clusters:

    def __init__(self):
        self.numero_clusters = 10                                       # Número de clusters para os algoritmos que utilizam desse atributo
        self.figura = 0                                                 # Gerencia o número de figuras    
        self.lista_corrigida = np.array(self.get_lista_corrigida())     # Lista com os valores corrigidos pelos especialistas

    def main(self):
        self.set_amostras('Tamura')
        self.analise('Tamura', 'AGNES')         # Alterar parametros para colher novos resultados
        # self.analise('Tamura', 'BIRCH')       # Arrumar os parametros
        # self.analise('Tamura', 'CLARANS')     # Arrumar os parametros
        self.analise('Tamura', 'CURE')          # Alterar parametros para colher novos resultados
        # self.analise('Tamura', 'DBSCAN')      # Arrumar os parametros
        self.analise('Tamura', 'FCM')           # Alterar parametros para colher novos resultados
        self.analise('Tamura', 'KMEANS')        # Alterar parametros para colher novos resultados
        self.analise('Tamura', 'KMEDOIDS')      # Alterar parametros para colher novos resultados
        # self.analise('Tamura', 'OPTICS')      # Arrumar os parametros
        # self.analise('Tamura', 'ROCK')        # Arrumar os parametros
        plt.show()

    def analise(self, descritor, algoritmo):
        lista_agrupada = self.get_modelo(algoritmo)
        if lista_agrupada is None:
            pass
        else:
            print('Grupos {}'.format(np.unique(lista_agrupada)))
            print(algoritmo+'-'+descritor+' Scores')
            print('Fowlkes-Mallows score: ' + str(metrics.fowlkes_mallows_score(self.lista_corrigida, lista_agrupada)))
            print('Davies-Bouldin score: ' + str(metrics.davies_bouldin_score(self.amostras, lista_agrupada)))
            print('Calinski and Harabasz score: '+str(metrics.calinski_harabasz_score(self.amostras, lista_agrupada)) + '\n\n')
            #self.visualiza_clusterizacao(descritor, lista_agrupada)'

    def set_amostras(self, descritor):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/' + descritor + '.txt')]
        aux = []
        for linha in linhas[1:]:                    # Skip primeira linha (Contem apenas metadata)
            auxList = linha.split(' ')
            auxList = list(filter(None, auxList))   # Remove espacos em branco
            auxList.pop(0)                          # Remove Id da amostra no arquivo
            auxList.pop(0)                          # Remove a classe
            auxList = list(map(float, auxList))
            auxList = np.array(auxList)
            aux.append(auxList)
        #self.amostras = np.array(aux)
        self.amostras = aux                         # Lista de lista

    def nova_figura(self):
        plt.figura(self.figura)
        self.figura = self.figura + 1

    def get_lista_corrigida(self):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/true_label.txt')]
        lista_corrigida = []
        for linha in linhas:
            lista_corrigida.append(linha)
        return lista_corrigida

    def get_modelo(self, algoritmo):
        instance = None

        if algoritmo == 'AGNES':
            instance = agglomerative(self.amostras, self.numero_clusters)                               # Alterar parametros para colher novos resultados
        elif algoritmo == 'BIRCH':
            instance = birch(self.amostras, self.numero_clusters)                                       # Arrumar os parametros
        elif algoritmo == 'CLARANS':
            instance = clarans(self.amostras, self.numero_clusters, numlocal=None, maxneighbor=None)    # Arrumar os parametros
        elif algoritmo == 'CURE':
            instance = cure(self.amostras, self.numero_clusters)                                        # Alterar parametros para colher novos resultados
        elif algoritmo == 'DBSCAN':
            instance = dbscan(self.amostras, eps=None, neighbors=None)                                  # Arrumar os parametros
        elif algoritmo == 'FCM':
            initial_centers = kmeans_plusplus_initializer(self.amostras, self.numero_clusters).initialize()  # Alterar parametros para colher novos resultados
            instance = fcm(self.amostras, initial_centers)
        elif algoritmo == 'KMEANS':
            initial_centers = kmeans_plusplus_initializer(self.amostras, self.numero_clusters).initialize()  # Alterar parametros para colher novos resultados
            instance = kmeans(self.amostras, initial_centers)
        elif algoritmo == 'KMEDOIDS':
            instance = kmedoids(self.amostras, initial_index_medoids=[0,100,200,300,400,500,600,700,800,900]) # Alterar parametros para colher novos resultados
        elif algoritmo == 'OPTICS':
            instance = optics(self.amostras, eps=None, minpts=None)                                     # Arrumar os parametros
        elif algoritmo == 'ROCK':
            instance = rock(self.amostras, eps=None, self.number_clusters)                              # Arrumar os parametros
        else:
            pass

        instance.process()
        lista_agrupada = self.get_lista_agrupada(instance.get_clusters())
        return np.array(lista_agrupada)

    def get_lista_agrupada(self, clusters):
        lista_aux = [0] * len(self.amostras)
        count = 0
        for cluster in clusters:
            for index in cluster:
                lista_aux[index] = count
            count = count + 1
        return lista_aux


if __name__ == '__main__':
    Clusters().main()
