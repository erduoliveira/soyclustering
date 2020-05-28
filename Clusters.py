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

from pyclustering.cluster import cluster_visualizer_multidim
from Utils import Utils

import xlwt

class Clusters:

    def __init__(self):
        self.util = Utils()
        self.figura = 0
        self.numero_clusters = 10
        self.descritores = ['BIC', 'CEDD', 'FCTH', 'Gabor', 'GCH',
               'Haralick', 'HaralickColor', 'HaralickFull', 'JCD',
               'LBP', 'LCH', 'Moments', 'MPO', 'MPOC',
               'PHOG', 'ReferenceColorSimilarity', 'Tamura']
        self.algoritmos = ['AGNES', 'CLARANS', 'CURE', 'DBSCAN',
                           'FCM', 'KMEANS', 'KMEDOIDS' 'OPTICS', 'ROCK']
        self.conjunto = 'caracteristicas-d6'
        self.lista_corrigida = np.array(self.util.get_lista_corrigida(self.conjunto))
        self.wb = xlwt.Workbook()
        
    def main(self):
        algoritmo = 'ROCK'
        descritor = 'Tamura'
        #for descritor in self.descritores:
        self.amostras = self.util.get_amostras(self.conjunto, descritor)
        self.lista_agrupada = self.get_modelo(algoritmo)
        self.analise(descritor,algoritmo)

    def grava_resultados(self, wb):
        for algoritmo in self.algoritmos:
            ws = self.wb.add_sheet(algoritmo)
            ws.write(0,0,'algoritmo')
            ws.write(0,1,'descritor')
            ws.write(0,2,'numero de grupos')
            ws.write(0,3,'fowkes-m')
            ws.write(0,4,'davies-b')
            ws.write(0,5,'calinski-h')
            l = 1
            for descritor in self.descritores:
                self.amostras = self.util.get_amostras(self.conjunto, descritor)
                self.lista_agrupada = self.get_modelo(algoritmo)
                self.save_analise(descritor, algoritmo, l, ws)
                l = l + 1
        self.wb.save('exp-anteriores/d6results.xls')

    def analise(self, descritor, algoritmo):
        if self.lista_agrupada is None:
            pass
        else:
            print('Grupos {}'.format(np.unique(self.lista_agrupada)))
            print(algoritmo+'-'+descritor+' Scores')
            print('Fowlkes-Mallows score: ' + str(metrics.fowlkes_mallows_score(self.lista_corrigida, self.lista_agrupada)))
            print('Davies-Bouldin score: ' + str(metrics.davies_bouldin_score(self.amostras, self.lista_agrupada)))
            print('Calinski and Harabasz score: '+str(metrics.calinski_harabasz_score(self.amostras, self.lista_agrupada)) + '\n\n')

    def save_analise(self, descritor, algoritmo, linha, ws):
        if self.lista_agrupada is None:
            pass
        else:
            try:
                ws.write(linha, 0, algoritmo)
                ws.write(linha, 1, descritor)
                ws.write(linha, 2, len(np.unique(self.lista_agrupada))) #numero de grupo
                ws.write(linha, 3, metrics.fowlkes_mallows_score(self.lista_corrigida, self.lista_agrupada))
                ws.write(linha, 4, metrics.davies_bouldin_score(self.amostras, self.lista_agrupada))
                ws.write(linha, 5, metrics.calinski_harabasz_score(self.amostras, self.lista_agrupada))
            except:
                print('Tive de pular a etapa -> '+algoritmo+ ' '+ descritor)
         
    def get_modelo(self, algoritmo):
        instance = None

        if algoritmo == 'AGNES':
            instance = agglomerative(self.amostras, self.numero_clusters, link=None)                               # Alterar parametros para colher novos resultados
        elif algoritmo == 'BIRCH':
            instance = birch(self.amostras, self.numero_clusters)                                       # Arrumar os parametros
        elif algoritmo == 'CLARANS':
            instance = clarans(self.amostras, self.numero_clusters, numlocal=100, maxneighbor=1)    # 500,1 talvez 200,1 ja funcione
        elif algoritmo == 'CURE':
            instance = cure(self.amostras, self.numero_clusters, number_represent_points=5, compression=0.5)                                        # Alterar parametros para colher novos resultados
        elif algoritmo == 'DBSCAN':
            instance = dbscan(self.amostras, eps=77.26, neighbors=8)                                  # Arrumar os parametros
        elif algoritmo == 'FCM':
            initial_centers = kmeans_plusplus_initializer(self.amostras, 10).initialize()  # Alterar parametros para colher novos resultados
            instance = fcm(self.amostras, initial_centers)
        elif algoritmo == 'KMEANS':
            initial_centers = kmeans_plusplus_initializer(self.amostras, self.numero_clusters).initialize()  # Alterar parametros para colher novos resultados
            instance = kmeans(self.amostras, initial_centers, tolerance=0.001)
        elif algoritmo == 'KMEDOIDS':
            instance = kmedoids(self.amostras, initial_index_medoids=[0,0,0,0,0,0,0,0,0,0], tolerance=0.0001) # Alterar parametros para colher novos resultados
        elif algoritmo == 'OPTICS':
            instance = optics(self.amostras, eps=1.74603, minpts=3)                                     # Arrumar os parametros
        elif algoritmo == 'ROCK':
            instance = rock(self.amostras, eps=461.4553, number_clusters=self.numero_clusters, threshold=0.5)                              # Arrumar os parametros
        else:
            pass

        instance.process()
        lista_agrupada = self.get_lista_agrupada(instance.get_clusters())
        return np.array(lista_agrupada)

    ##
    # Diferente de outros frameworks o pyclustering gera um resultado
    # em que são formados x grupos, em que cada grupo corresponde a uma classe
    # estou trabalhando com ordem das amostras, logo se faz necessario alterar
    # as posicoes na lista
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
