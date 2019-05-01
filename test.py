from main import parse
from main import bootstrap
import decisionTree as dt

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

def testBootstrap(dataPanda):
    print("\nBOOTSTRAP 1:")
    (treino, teste) = bootstrap(dataPanda)
    print("treino: "+ str(len(treino)) + "\n" + str(treino))
    print("teste: "+ str(len(teste)) + "\n" + str(teste))
    print("\nBOOTSTRAP 2:")
    (treino, teste) = bootstrap(dataPanda)
    print("treino: "+ str(len(treino)) + "\n" + str(treino))
    print("teste: "+ str(len(teste)) + "\n" + str(teste))

def testInduce(dataPanda):
    root = dt.DecisionTree()
    root.makeRootNode(dataPanda)
    root.induce(root.data, root.listOfAttr)
    root._print(0)
    return root

def test(dataPanda):
    tree = testInduce(dataPanda)
    testClassify(tree)
    testBootstrap(dataPanda)