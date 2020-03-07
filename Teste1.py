
class Teste1:

    def __init__(self):
        self.training_file = 'bic-training/BIC.txt'
        self.samples = []

    def main(self):
        self.samples = self.get_samples()
        print(self.samples)

    def get_samples(self):
        samples = []
        lineList = [line.rstrip('\n') for line in open(self.training_file)]
        for line in lineList:
            samples.append(line.split(' '))

        for sample in samples:
            sample.pop()
            sample.pop()
            sample.pop(0)
            sample.pop(0)
        
        return samples


if __name__ == '__main__':
    Teste1().main()