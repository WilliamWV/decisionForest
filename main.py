import decisionTree as dt
import pandas as pd

# Quantidade de árvores na floresta
ntree  = 10

#Floresta de árvores de decisão
forest = []

def parse(fileName):
    data = pd.read_csv(fileName, delimiter=';', header=0)
    dataPanda = data
    dt.Data = data.values.tolist()
    return dataPanda

def main():
    fileName = "dadosBenchmark_validacaoAlgoritmoAD.csv"
    dataPanda = parse(fileName)
    root = dt.DecisionTree()
    root.makeRootNode(dataPanda)
    root.induce(root.data, root.listOfAttr)
    root._print(0)

if __name__ == "__main__":
    main()