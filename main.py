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


def testClassify(tree):
    #Instâncias já existentes
    inst1 = ["Ensolarado", "Quente", "Alta", "Falso"] # Esperar Nao
    inst2 = ["Ensolarado", "Fria", "Normal", "Falso"] # Esperar Sim
    inst3 = ["Chuvoso", "Fria", "Normal", "Verdadeiro"] # Esperar Nao
    #Instâncias novas
    inst4 = ["Ensolarado", "Fria", "Alta", "Verdadeiro"] 
    inst5 = ["Nublado", "Amena", "Normal", "Falso"]
    inst6 = ["Chuvoso", "Quente", "Alta", "Falso"]
    
    insts = [inst1, inst2, inst3, inst4, inst5, inst6]

    for inst in insts:
        print ("Classificação da instância " + str(inst) + ":")
        ans = tree.classify(inst)
        print ("Resultado da classificação: " + str(ans))


def main():
    fileName = "dadosBenchmark_validacaoAlgoritmoAD.csv"
    dataPanda = parse(fileName)
    root = dt.DecisionTree()
    root.makeRootNode(dataPanda)
    root.induce(root.data, root.listOfAttr)
    root._print(0)
    testClassify(root)
    

if __name__ == "__main__":
    main()