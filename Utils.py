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
        self.descritores = ['BIC', 'CEDD', 'FCTH', 'Gabor', 'GCH',
               'Haralick', 'HaralickColor', 'HaralickFull', 'JCD',
               'LBP', 'LCH', 'Moments', 'MPO', 'MPOC',
               'PHOG', 'ReferenceColorSimilarity', 'Tamura']
        self.figura = 0
        self.numero_clusters = 10

    def main(self):
        self.get_visualizacoes()

    def get_visualizacoes(self):
        for descritor in self.descritores:
            self.figura = self.nova_figura(self.figura)
            amostras = self.get_amostras('caracteristicas-d6', descritor)
            self.visualizacao_scatter(amostras, descritor, 'isomap')
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

    def visualiza_clusterizacao(self, amostras, pred_label, algoritmo, descritor):
        pca = PCA(n_components=2)
        visualizacao = pca.fit_transform(amostras)
        plt.close()
        self.nova_figura(self.figura)
        plt.figure(figsize=(16, 12))
        numero_cluster = len(np.unique(pred_label))
        plt.suptitle(algoritmo + ' - ' + descritor + ' (PCA)')
        ax = sns.scatterplot(x=visualizacao[:, 0],
                             y=visualizacao[:, 1],
                             hue=pred_label,
                             palette=sns.color_palette('Set1', n_colors=numero_cluster)
                             )
        ax.get_legend().remove()
        #plt.show()
        plt.savefig('plots\\ag-d8\\'+algoritmo+'_'+descritor, dpi=500)

    def nova_figura(self, figura):
        plt.figure(figura)
        return figura + 1

    def get_amostras(self, conjunto, descritor):
        linhas = [linha.rstrip('\n') for linha in open(conjunto+'/' + descritor + '.txt')]
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
        linhas = [linha.rstrip('\n') for linha in open(conjunto+'/true_label.txt')]
        lista_corrigida = []
        for linha in linhas:
            lista_corrigida.append(linha)
        return lista_corrigida


if __name__ == '__main__':
    Utils().main()