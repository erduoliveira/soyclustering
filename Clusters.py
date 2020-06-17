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
from pyclustering.cluster.hsyncnet import hsyncnet
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
        self.descritores = ['BIC', 'CEDD', 'FCTH', 'Gabor', 'GCH',
               'Haralick', 'HaralickColor', 'HaralickFull', 'JCD',
               'LBP', 'LCH', 'Moments', 'MPO', 'MPOC',
               'PHOG', 'ReferenceColorSimilarity', 'Tamura']
        self.algoritmos = ['AGNES', 'CLARANS', 'CURE',
                           'FCM', 'KMEANS', 'KMEDOIDS']
        self.algoritmos_param = ['DBSCAN', 'OPTICS', 'ROCK']
        self.dbscan_param = [[18950,2], [3.1,12], [2,2], [0.34,2], [33853.6,3],
                             [77.26,8], [844.66,5], [95193400000,3], [2.5,2], [3060,2],
                             [22866.1,2], [1.443,3], [31.159,5], [219.62,2], [0.0862939,2],
                             [0.0358037,4], [110.98,2]]
        # self.optics_param = [[22929.5,2], [3.1,3], [2,2], [0.4114,2], [40962.856,2],
        #                      [93.4846,5], [1022.0386,3], [115184014000,3], [3.025,4], [3702.6,5],
        #                      [27677.981,5], [1.74603,3], [37.70239,4], [265.7402,3], [0.104415619,9],
        #                      [0.043322477,2], [134.2858,1]]
        self.optics_param = [[18950,2], [3.1,3], [2,2], [0.34,2], [33853.6,2],
                             [77.26,5], [844.66,3], [95193400000,3], [2.5,4], [3060,5],
                             [22866.1,5], [1.443,3], [31.159,4], [219.62,3], [0.0862939,2],
                             [0.0358037,2], [110.98,1]]
        self.rock_param = [[77000,0], [12.89,0], [8.32,0], [1.41,0], [140763,0], [321.25,0],
                           [3512.1,0], [395814553014,0], [10.40,0], [12723.49,0], [95077.34,0], [6,0], [129.56,0],
                           [1100,0], [0.36,0], [0.15,0], [461.46,0]]
        self.conjunto = 'caracteristicas-d8'
        self.numero_clusters = 10
        self.lista_corrigida = np.array(self.util.get_lista_corrigida(self.conjunto))
        self.wb = xlwt.Workbook()
        
    def main(self):
        algoritmo = 'ROCK'
        descritor = 'JCD'
        self.amostras = self.util.get_amostras(self.conjunto, descritor)
        self.lista_agrupada = self.get_modelo(algoritmo, 12.99675, 0)
        self.analise(descritor, algoritmo)
        self.util.visualiza_clusterizacao(self.amostras, self.lista_agrupada, algoritmo, descritor)
        # parametros = [self.dbscan_param, self.optics_param, self.rock_param]
        # i = 0
        # for algoritmo in self.algoritmos_param:
        #     j = 0
        #     for descritor in self.descritores:
        #         self.amostras = self.util.get_amostras(self.conjunto, descritor)
        #         #self.lista_agrupada = self.get_modelo(algoritmo, 0, 0)
        #         self.lista_agrupada = self.get_modelo(algoritmo, parametros[i][j][0], parametros[i][j][1])
        #         self.analise(descritor, algoritmo)
        #         self.util.visualiza_clusterizacao(self.amostras, self.lista_agrupada, algoritmo, descritor)
        #         j = j + 1
        #     i = i + 1
        #self.grava_resultados()
        #self.save_algoritmos_param()

    def save_algoritmos_param(self):
        parametros = [self.dbscan_param, self.optics_param, self.rock_param]
        i = 0
        for algoritmo in self.algoritmos_param:
            j = 0
            ws = self.wb.add_sheet(algoritmo)
            ws.write(0, 0, 'algoritmo')
            ws.write(0, 1, 'descritor')
            ws.write(0, 2, 'numero de grupos')
            ws.write(0, 3, 'fowkes-m')
            ws.write(0, 4, 'davies-b')
            ws.write(0, 5, 'calinski-h')
            for descritor in self.descritores:
                self.amostras = self.util.get_amostras(self.conjunto, descritor)
                self.lista_agrupada = self.get_modelo(algoritmo, parametros[i][j][0], parametros[i][j][1])
                self.save_analise(descritor, algoritmo, j+1, ws)
                j = j + 1
            i = i + 1
        self.wb.save('exp-anteriores/d771results.xls')

    def desviop(self, range, algoritmo):
        for descritor in self.descritores:
            self.amostras = self.util.get_amostras(self.conjunto, descritor)
            ws = self.wb.add_sheet(descritor)
            ws.write(0, 0, 'algoritmo')
            ws.write(0, 1, 'descritor')
            ws.write(0, 2, 'numero de grupos')
            ws.write(0, 3, 'fowkes-m')
            ws.write(0, 4, 'davies-b')
            ws.write(0, 5, 'calinski-h')
            for l in range(1,range):
                self.lista_agrupada = self.get_modelo(algoritmo)
                self.save_analise(descritor, algoritmo, l, ws)
        self.wb.save('exp-anteriores/'+algoritmo+'-d7.xls')

    def grava_resultados(self):
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
        self.wb.save('exp-anteriores/d71results.xls')

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
                exit(-1)
         
    def get_modelo(self, algoritmo, eps, neig):
        # eps = 0
        # neig = 0
        print(algoritmo+ ' '+ str(eps)+ ' - '+ str(neig))
        instance = None

        if algoritmo == 'AGNES':
            instance = agglomerative(self.amostras, self.numero_clusters, link=None)
        elif algoritmo == 'BIRCH':
            instance = birch(self.amostras, self.numero_clusters, entry_size_limit=10000)
        elif algoritmo == 'CLARANS':
            instance = clarans(self.amostras, self.numero_clusters, numlocal=100, maxneighbor=1)
        elif algoritmo == 'CURE':
            instance = cure(self.amostras, self.numero_clusters, number_represent_points=5, compression=0.5)
        elif algoritmo == 'DBSCAN':
            instance = dbscan(self.amostras, eps=eps, neighbors=neig)
        elif algoritmo == 'FCM':
            initial_centers = kmeans_plusplus_initializer(self.amostras, self.numero_clusters).initialize()
            instance = fcm(self.amostras, initial_centers)
        elif algoritmo == 'KMEANS':
            initial_centers = kmeans_plusplus_initializer(self.amostras, self.numero_clusters).initialize()
            instance = kmeans(self.amostras, initial_centers, tolerance=0.001)
        elif algoritmo == 'KMEDOIDS':
            instance = kmedoids(self.amostras, initial_index_medoids=[0,0,0,0,0,0,0,0,0,0], tolerance=0.0001) #ajustar o n_de cluster
        elif algoritmo == 'OPTICS':
            instance = optics(self.amostras, eps=eps, minpts=neig)
        elif algoritmo == 'ROCK':
            instance = rock(self.amostras, eps=eps, number_clusters=self.numero_clusters, threshold=0.5)
        else:
            pass

        instance.process()
        lista_agrupada = self.get_lista_agrupada(instance.get_clusters())
        lista_agrupada = np.array(lista_agrupada)
        # n_grupos = len(np.unique(lista_agrupada))
        # if n_grupos > 7:
        #     lista_agrupada = self.get_modelo(algoritmo, eps, neig+1)
        return lista_agrupada

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
