from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

class Teste1:

    def __init__(self):
        self.training_file = 'bic-training/BIC.txt'

    def main(self):)
        self.amostras = np.array(self.get_samples())
        print(a.shape)
        modelo = KMeans(n_clusters=4)
        modelo.fit(self.amostras)
        print('Grupos {}' .format(modelo.labels_))


    def get_samples(self):
        linhas = [linha.rstrip('\n') for linha in open(self.training_file)]
        aux = []
        for linha in linhas[1:]: #skip primeira linha
            auxList = linha.split(' ')
            auxList.pop()
            auxList.pop()
            auxList.pop(0)
            auxList.pop(0)
            auxList = list(map(float, auxList))
            auxList = np.array(auxList)
            aux.append(auxList)
        return aux


if __name__ == '__main__':
    Teste1().main()