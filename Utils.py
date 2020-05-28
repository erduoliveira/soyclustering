import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


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
            amostras = self.get_amostras('caracteristicas-d8', descritor)
            self.visualizacao_pca(amostras, descritor)
        plt.show()

    def visualizacao_pca(self, amostras, descritor):
        pca = PCA(n_components=2)
        visualizacao = pca.fit_transform(amostras)
        plt.suptitle('1ª e 2ª aquisicao site-corrigida - Descritor '+ descritor)
        ax = sns.scatterplot(
            x=visualizacao[:,0],
            y=visualizacao[:,1],
        )

    def visualizacao_truncatedSVD(self, amostras, descritor):
        svd = TruncatedSVD(n_components=2)
        visualizacao = svd.fit_transform(amostras)
        plt.suptitle('2ª aquisicao site-corrigida - Descritor '+ descritor)
        ax = sns.scatterplot(
            x=visualizacao[:,0],
            y=visualizacao[:,1],
        )

    def visualiza_clusterizacao(self, amostras, pred_label, algoritmo, descritor):
        pca = PCA(n_components=2)
        visualizacao = pca.fit_transform(amostras)
        self.nova_figura(self.figura)
        plt.suptitle(algoritmo + ' - ' + descritor)
        # plt.subplot(1, 2, 1)
        ax = sns.scatterplot(x=visualizacao[:, 0],
                             y=visualizacao[:, 1],
                             hue=pred_label,
                             palette=sns.color_palette('Set1', 7)
                             #palette=sns.color_palette('Set1', self.numero_clusters)
                             )
        ax.get_legend().remove()
        # ax.set_title('True label')
        # plt.subplot(1, 2, 2)
        # ax = sns.scatterplot(x=visualizacao[:, 0],
        #                      y=visualizacao[:, 1],
        #                      hue=pred_label,
        #                      palette=sns.color_palette('Set1', self.numero_clusters)
        #                      )
        # ax.get_legend().remove()
        # ax.set_title('Pred label')
        plt.show()

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