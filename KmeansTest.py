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
        self.numero_clusters = 7
        self.figure = 0
        self.true_label = np.array(self.get_true_label())  # true label para o dataset escolhido no caso o D6

    def main(self):
        # self.show_elbow()
        # self.show_silhouette()
        self.analise('BIC')
        self.analise('FCTH')
        self.analise('Gabor')
        plt.show()

    def analise(self, descritor):
        self.set_amostras(descritor)
        modelo = KMeans(n_clusters=self.numero_clusters).fit(self.amostras)
        pred_label = modelo.labels_
        print(descritor + ' Scores')
        print('Fowlkes-Mallows score: ' + str(metrics.fowlkes_mallows_score(self.true_label, pred_label)))
        print('Davies-Bouldin score: ' + str(metrics.davies_bouldin_score(self.amostras, pred_label)))
        print(
            'Calinski and Harabasz score: ' + str(metrics.calinski_harabasz_score(self.amostras, pred_label)) + '\n\n')
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

    def show_silhouette(self):
        self.set_amostras('BIC')
        self.get_silhouette('BIC')
        self.set_amostras('FCTH')
        self.get_silhouette('FCTH')
        self.set_amostras('Gabor')
        self.get_silhouette('Gabor')

    def get_silhouette(self, descritor):
        self.nova_figura()
        silhouette = [self.silhouette_kmeans(numero_clusters) for numero_clusters in range(2, 50)]
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

    def show_elbow(self):
        self.set_amostras('BIC')
        self.get_elbow('BIC')
        self.set_amostras('FCTH')
        self.get_elbow('FCTH')
        self.set_amostras('Gabor')
        self.get_elbow('GABOR')

    def set_amostras(self, descritor):
        self.amostras = np.array(self.get_samples(descritor))

    def get_elbow(self, descritor):
        self.nova_figura()
        elbow = [self.elbow_kmeans(numero_clusters) for numero_clusters in range(1, 50)]
        elbow = pd.DataFrame(elbow, columns=['grupos', 'inertia'])
        plt.plot(elbow.grupos, elbow.inertia)
        plt.suptitle('Elbow method - ' + descritor)
        plt.xlabel('Numero de clusters')
        plt.ylabel('SSE')
        plt.xticks(elbow.grupos)
        plt.ylim(min(elbow.inertia), elbow.inertia[1])
        plt.xlim(2, 50)

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
        return aux

    def nova_figura(self):
        plt.figure(self.figure)
        self.figure = self.figure + 1

    def get_true_label(self):
        true_label = [0] * 128  # OX
        true_label[128:136] = [1] * 9  # 2M
        true_label[137:253] = [2] * 117  # 2P
        true_label[254:456] = [3] * 203  # 2U
        true_label[457:480] = [4] * 24  # 3M
        true_label[481:622] = [5] * 142  # 3P
        true_label[623:653] = [6] * 31  # 3U
        return true_label


if __name__ == '__main__':
    KmeansTest().main()
