import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.transforms import Bbox
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import umap


class Utils():

    def __init__(self):
        # self.descritores = ['BIC', 'CEDD', 'FCTH', 'Gabor', 'GCH',
        #        'Haralick', 'HaralickColor', 'HaralickFull', 'JCD',
        #        'LBP', 'LCH', 'Moments', 'MPO', 'MPOC',
        #        'PHOG', 'ReferenceColorSimilarity', 'Tamura']
        self.figura = 0
        # self.numero_clusters = 10
        # self.conjunto = 'caracteristicas-d6'
        # self.lista_corrigida = self.get_lista_corrigida(self.conjunto)
        self.makers_utilizados = ('o', 'd', '^', '<', '>', 's', 'P', 'X','v','D')

    def main(self):
        self.get_visualizacoes()

    def get_visualizacoes(self):
        #for descritor in self.descritores:
        self.figura = self.nova_figura(self.figura)
        amostras = self.get_amostras(self.conjunto, 'MPO')
        self.visualiza_clusterizacao(amostras,self.lista_corrigida,'Rotulo Verdadeiro', 'MPO')
        #plt.show()

    def visualizacao_scatter(self, amostras, descritor, tecnica_reducao):
        if tecnica_reducao == 'pca':
            redutor = PCA(n_components=2)
        elif tecnica_reducao == 'tsvd':
            redutor = TruncatedSVD(n_components=2)
        elif tecnica_reducao == 'fa':
            redutor = FactorAnalysis(n_components=2)
        elif tecnica_reducao == 'ica':
            redutor = FastICA(n_components=2, random_state=12)
        elif tecnica_reducao == 'tsne':
            redutor = TSNE(n_components=2)
        elif tecnica_reducao == 'isomap':
            redutor = Isomap(n_components=2, n_neighbors=5)
        plt.figure(figsize=(16, 12))
        visualizacao = redutor.fit_transform(amostras)
        plt.suptitle('1ª aquisicao site-corrigida - '+tecnica_reducao+' - '+ descritor)
        ax = sns.scatterplot(
            x=visualizacao[:,0],
            y=visualizacao[:,1],
        )
        plt.savefig('d6-plt-'+tecnica_reducao+'\\'+descritor, dpi=500)

    def visualiza_clusterizacao(self, amostras, lista_corrigida, pred_label, algoritmo, descritor, path, metodo):
        inst = None
        if metodo == 'TSNE':
            inst = TSNE(n_components=2)
        elif metodo == 'PCA':
            inst = PCA(n_components=2)
        else:
            print('Metodo de reducao nao encontrado: '+ metodo)
            exit(-1)
        visualizacao = inst.fit_transform(amostras)
        plt.close()
        self.nova_figura(self.figura)
        plt.figure(figsize=(16, 12))
        numero_cluster = len(np.unique(pred_label))
        plt.suptitle(algoritmo + ' - ' + descritor + ' ' + metodo)
        ax = sns.scatterplot(x=visualizacao[:, 0],
                             y=visualizacao[:, 1],
                             hue=pred_label,
                             palette=sns.color_palette(palette=None, n_colors=numero_cluster, desat=1),
                             markers=self.makers_utilizados,
                             style=lista_corrigida,
                             s=100
                             )
        ax.get_legend().remove()
        plt.savefig(path+metodo+'_'+descritor+'_'+algoritmo, dpi=200)
        plt.close()

    def nova_figura(self, figura):
        plt.figure(figura)
        return figura + 1

    def get_amostras(self, conjunto, descritor):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/v2/'+conjunto+'/' + descritor + '.txt')]
        aux = []
        for linha in linhas[1:]:                  # Skip primeira linha (Contem apenas metadata)
            auxList = linha.split(' ')
            auxList = list(filter(None, auxList))   # Remove espacos em branco
            auxList.pop(0)                          # Remove Id da amostra no arquivo
            auxList.pop(0)                          # Remove a classe
            auxList = list(map(float, auxList))
            auxList = np.array(auxList)
            aux.append(auxList)
        return aux

    def get_lista_corrigida(self, conjunto):
        linhas = [linha.rstrip('\n') for linha in open('caracteristicas/v2/'+conjunto+'/true_label.txt')]
        lista_corrigida = []
        for linha in linhas:
            lista_corrigida.append(linha)
        return lista_corrigida

    def get_densidade_param(self, conjunto):
        if conjunto == 'd6':
            dbscan_param = [[0.304499,2], [13330, 2], [3.017, 1], [1.4142, 4], [0.197662, 1], [23902.4, 2],
                             [47.2569, 14], [848.725, 3], [54711800000, 13], [2.55, 1], [2346.05, 2],
                             [23234, 2], [1.6625, 3], [24.514, 8], [262, 1], [0.086381, 1],
                             [0.03085, 3], [94.86, 1]]
            optics_param = [[0.304499,2], [13330, 2], [3.017, 1], [1.4142, 4], [0.197662, 1], [23902.4, 2],
                             [47.2569, 14], [848.725, 3], [54711800000, 13], [2.55, 1], [2346.05, 2],
                             [23234, 2], [1.6625, 3], [24.514, 8], [262, 1], [0.086381, 1],
                             [0.03085, 3], [94.86, 1]]
            rock_param = [[1.5224, 0], [66650,0],[15.0869, 0], [7.07, 0], [0.98831, 0], [119512, 0], [236.28, 0],
                          [4243.625, 0], [273559000000, 0], [12.75, 0], [11730, 0], [116170, 0], [8.31, 0], [122.572, 0],
                          [1310, 0], [0.4319, 0], [0.15425, 0], [474.336, 0]]
            return dbscan_param, optics_param, rock_param
        elif conjunto == 'd7':
            dbscan_param = [[0.445597,2], [17055, 2], [2.79, 12], [1.8, 1], [0.306, 2], [30468.24, 3],
                            [69.534, 11], [760.194, 5], [85674060000, 3], [2.25, 2], [2754, 2],
                            [20579.49, 4], [1.2987, 3], [28.0431, 13], [197.658, 2], [0.07766, 2],
                            [0.03222, 4], [99.882, 1]]
            optics_param = [[0.445597,2], [17055, 2], [2.79, 5], [1.8, 1], [0.306, 2], [30468.24, 2],
                            [69.534, 11], [760.194, 3], [85674060000, 3], [2.25, 4], [2754, 5],
                            [20579.49, 5], [1.2987, 3], [28.0431, 13], [197.658, 3], [0.07766, 9],
                            [0.03222, 2], [99.882, 1]]
            rock_param = [[2.23,1], [85275, 0], [13.95, 0], [9, 0], [1.53, 0], [152341.2, 0], [347.67, 0],
                          [3800.97, 0], [428370300000, 0], [11.25, 0], [13770, 0], [13770, 0], [6.49, 0], [140.2155, 0],
                          [988.29, 0], [0.38832, 0], [0.16112, 0], [499.41, 0]]
            return dbscan_param, optics_param, rock_param
        elif conjunto == 'd8':
            dbscan_param = [[0.3499,2], [16740,1], [2.83182,1], [1.4172,2], [0.13661,6], [33755,1],
                            [29.4915,16], [742.7,4], [32017400000,17], [2.3451,1], [3383.2,1],
                            [30637,1], [1.14858,3], [15.853,19], [231.836,2], [0.0844045,1],
                            [0.0372035,1], [84.3817,1]]
            optics_param = [[0.3499,1], [16740,1], [2.83182,1], [1.4172,1], [0.13661,1], [33755,1],
                            [29.4915,1], [742.7,1], [32017400000,1], [2.3451,1], [3383.2,1],
                            [30637,1], [1.14858,1], [15.853,1], [231.836,1], [0.0844045,1],
                            [0.0372035,1], [84.3817,1]]
            rock_param = [[1.7495,0], [83700, 0], [14.1591, 0], [7.086, 0], [0.68305, 0], [168777, 0], [147.457, 0],
                          [3713.515, 0], [160100000000, 0], [11.725, 0], [16916, 0], [153185, 0], [5.7429, 0], [79.2665, 0],
                          [1159.18, 0], [0.4220225, 0], [0.1860175, 0], [421.9085, 0]]
            return dbscan_param, optics_param, rock_param
        elif conjunto == 'd9':
            dbscan_param = [[0.3499,2], [16740,1], [2.83182,1], [1.4172,2], [0.13661,16], [33755,2],
                            [29.4915,17], [742.7,4], [32017400000,22], [2.3451,1], [3383.2,1],
                            [30637,1], [1.14858,17], [15.853,21], [231.836,3], [0.0844045,1],
                            [0.0372035,5], [84.3817,1]]
            optics_param = [[0.3499,2], [16740,1], [2.83182,1], [1.4172,2], [0.13661,16], [33755,2],
                            [29.4915,17], [742.7,4], [32017400000,22], [2.3451,1], [3383.2,1],
                            [30637,1], [1.14858,17], [15.853,21], [231.836,3], [0.0844045,1],
                            [0.0372035,5], [84.3817,1]]
            rock_param = [[1.7495,0], [83700, 0], [14.1591, 0], [7.086, 0], [0.68305, 0], [168777, 0], [147.457, 0],
                          [3713.515, 0], [160100000000, 0], [11.725, 0], [16916, 0], [153185, 0], [5.7429, 0], [79.2665, 0],
                          [1159.18, 0], [0.4220225, 0], [0.1860175, 0], [421.9085, 0]]
            return dbscan_param, optics_param, rock_param
        elif conjunto == 'd10':
            dbscan_param = [[0.3499,3], [16740,3], [2.83182,2], [1.4172,3], [0.14,32], [33755,3],
                            [29.4915,20], [742.7,22], [32017400000,32], [3,1], [3383.2,2],
                            [34000,1], [1.14858,19], [15.853,25], [231.836,6], [0.0844045,2],
                            [0.0372035,14], [84.3817,2]]
            optics_param = [[0.3499,3], [16740,3], [2.83182,2], [1.4172,3], [0.14,32], [33755,1],
                            [29.4915,1], [742.7,1], [32017400000,1], [3,1], [3383.2,1],
                            [34000,1], [1.14858,1], [15.853,1], [231.836,1], [0.0844045,1],
                            [0.0372035,1], [84.3817,1]]
            rock_param = [[1.7495,0], [83700, 0], [14.1591, 0], [7.086, 0], [0.68305, 0], [168777, 0], [147.457, 0],
                          [3713.515, 0], [160100000000, 0], [11.725, 0], [16916, 0], [153185, 0], [5.7429, 0], [79.2665, 0],
                          [1159.18, 0], [0.4220225, 0], [0.1860175, 0], [421.9085, 0]]
            return dbscan_param, optics_param, rock_param
        else:
            print('Conjunto nao reconhecido')
            exit(-1)


if __name__ == '__main__':
    Utils().main()