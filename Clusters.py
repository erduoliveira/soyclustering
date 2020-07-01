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

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment


import xlwt

class Clusters:

    def __init__(self):
        self.util = Utils()
        self.figura = 0
        self.descritores = ['AutoColorCorrelogram', 'BIC', 'CEDD', 'FCTH',
               'Gabor', 'GCH', 'Haralick', 'HaralickColor', 'HaralickFull',
               'JCD', 'LBP', 'LCH', 'Moments', 'MPO', 'MPOC',
               'PHOG', 'ReferenceColorSimilarity', 'Tamura']
        self.algoritmos = ['AGNES', 'CURE',
                          'FCM', 'KMEANS', 'KMEDOIDS']
        #self.algoritmos = ['KMEDOIDS']
        self.algoritmos_param = ['DBSCAN', 'OPTICS', 'ROCK']
        self.conjunto = 'd7'
        # atualizar k-medoids iniciais ao trocar o numero de clusters
        self.numero_clusters = 7
        self.dbscan_param, self.optics_param, self.rock_param = self.util.get_densidade_param(self.conjunto)
        self.lista_corrigida = np.array(self.util.get_lista_corrigida(self.conjunto))
        self.wb = xlwt.Workbook()
        
    def main(self):
        # for descritor in self.descritores:
        #     self.amostras = self.util.get_amostras(self.conjunto, descritor)
        #     self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_corrigida,
        #                                       'True_label', descritor, 'plots/v2/' + self.conjunto + '/TRUE/', 'TSNE')
        #     self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_corrigida,
        #                                       'True_label', descritor, 'plots/v2/' + self.conjunto + '/TRUE/', 'PCA')
        # self.amostras = self.util.get_amostras(self.conjunto, descritor)
        # self.lista_agrupada = self.get_modelo(algoritmo, 0, 0)
        # print(self.acuracia(self.lista_corrigida,self.lista_agrupada))
        # self.analise(descritor, algoritmo)
        # self.util.visualiza_clusterizacao(self.amostras, self.lista_agrupada, algoritmo, descritor)
        self.grava_resultados('d7-testes1')
        #self.save_algoritmos_param('d7-densidade-autocolor')
        #self.desviop(51,'FCM')
        #self.desviop(51,'KMEANS')


    def acuracia(self, lista_corrigida, lista_agrupada):
        agrupada = lista_agrupada.astype(np.int)
        corrigida = lista_corrigida.astype(np.int)

        cm = confusion_matrix(corrigida, agrupada)
        indexes = np.asarray(linear_assignment(self.make_cost_m(cm))) #encontra a melhor ordem da matrix de confusao
        cm2 = cm[:, indexes[1]]                                       #altera a ordem na matrix de confusao
        #ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        #self.util.nova_figura(2)
        #ay = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
        #plt.show()
        # print(np.trace(cm2))
        # print(np.sum(cm2))
        return (np.trace(cm2)/np.sum(cm2))

    def make_cost_m(self, cm):
        s = np.max(cm)
        return (- cm + s)

    def save_algoritmos_param(self, nome_arquivo):
        parametros = [self.dbscan_param, self.optics_param, self.rock_param]
        i = 0
        for algoritmo in self.algoritmos_param:
            j = 0
            ws = self.wb.add_sheet(algoritmo)
            ws.write(0, 0, 'algoritmo')
            ws.write(0, 1, 'descritor')
            ws.write(0, 2, 'numero de grupos')
            ws.write(0, 3, 'acuracia')
            ws.write(0, 4, 'fowkes-m')
            ws.write(0, 5, 'davies-b')
            ws.write(0, 6, 'calinski-h')
            for descritor in self.descritores:
                self.amostras = self.util.get_amostras(self.conjunto, descritor)
                self.lista_agrupada = self.get_modelo(algoritmo, parametros[i][j][0], parametros[i][j][1])
                self.save_analise(descritor, algoritmo, j+1, ws)
                self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_agrupada,
                                                  algoritmo, descritor, 'plots/v2/'+self.conjunto+'/', 'TSNE')
                self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_agrupada,
                                                  algoritmo, descritor, 'plots/v2/'+self.conjunto+'/', 'PCA')
                j = j + 1
            i = i + 1
        self.wb.save('exp-anteriores/v2/'+self.conjunto+'/'+nome_arquivo+'.xls')

    def desviop(self, ntestes, algoritmo):
        for descritor in self.descritores:
            self.amostras = self.util.get_amostras(self.conjunto, descritor)
            ws = self.wb.add_sheet(descritor)
            ws.write(0, 0, 'algoritmo')
            ws.write(0, 1, 'descritor')
            ws.write(0, 2, 'numero de grupos')
            ws.write(0, 3, 'acuracia')
            ws.write(0, 4, 'fowkes-m')
            ws.write(0, 5, 'davies-b')
            ws.write(0, 6, 'calinski-h')
            for l in range(1,ntestes):
                self.lista_agrupada = self.get_modelo(algoritmo,0,0)
                self.save_analise(descritor, algoritmo, l, ws)
        self.wb.save('exp-anteriores/v2/'+self.conjunto+'/'+algoritmo+'.xls')

    def grava_resultados(self, nome_arquivo):
        for algoritmo in self.algoritmos:
            ws = self.wb.add_sheet(algoritmo)
            ws.write(0,0,'algoritmo')
            ws.write(0,1,'descritor')
            ws.write(0,2,'numero de grupos')
            ws.write(0,3,'acuracia')
            ws.write(0,4,'fowkes-m')
            ws.write(0,5,'davies-b')
            ws.write(0,6,'calinski-h')
            l = 1
            for descritor in self.descritores:
                self.amostras = self.util.get_amostras(self.conjunto, descritor)
                self.lista_agrupada = self.get_modelo(algoritmo,0,0)
                self.save_analise(descritor, algoritmo, l, ws)
                self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_agrupada,
                                                  algoritmo, descritor, 'plots/v2/' + self.conjunto + '/', 'TSNE')
                self.util.visualiza_clusterizacao(self.amostras, self.lista_corrigida, self.lista_agrupada,
                                                  algoritmo, descritor, 'plots/v2/' + self.conjunto + '/', 'PCA')
                l = l + 1
        self.wb.save('exp-anteriores/v2/'+self.conjunto+'/'+nome_arquivo+'.xls')

    def analise(self, descritor, algoritmo):
        if self.lista_agrupada is None:
            pass
        else:
            print('Grupos {}'.format(np.unique(self.lista_agrupada)))
            print(algoritmo+'-'+descritor+' Scores')
            print('Acuracia: ' + self.acuracia(self.lista_corrigida, self.lista_agrupada))
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
                ws.write(linha, 3, self.acuracia(self.lista_corrigida, self.lista_agrupada))
                ws.write(linha, 4, metrics.fowlkes_mallows_score(self.lista_corrigida, self.lista_agrupada))
                ws.write(linha, 5, metrics.davies_bouldin_score(self.amostras, self.lista_agrupada))
                ws.write(linha, 6, metrics.calinski_harabasz_score(self.amostras, self.lista_agrupada))
            except:
                print('Tive de pular a etapa -> '+algoritmo+ ' '+ descritor)
                exit(-1)
         
    def get_modelo(self, algoritmo, eps, neig):
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
            instance = kmedoids(self.amostras, initial_index_medoids=[0,0,0,0,0,0,0], tolerance=0.0001) #ajustar o n_de cluster
        elif algoritmo == 'OPTICS':
            instance = optics(self.amostras, eps=eps, minpts=neig)
        elif algoritmo == 'ROCK':
            instance = rock(self.amostras, eps=eps, number_clusters=self.numero_clusters, threshold=0.5)
        else:
            pass

        instance.process()
        lista_agrupada = self.get_lista_agrupada(instance.get_clusters())
        lista_agrupada = np.array(lista_agrupada)

        if (neig != 0):
            n_grupos = len(np.unique(lista_agrupada))
            if n_grupos > self.numero_clusters:
                lista_agrupada = self.get_modelo(algoritmo, eps, neig+1)
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
