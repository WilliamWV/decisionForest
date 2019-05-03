from main import parse
from main import bootstrap
import decisiontree as dt
import plottree as pt

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

def testBootstrap(data):
    print("\nBOOTSTRAP 1:")
    (treino, teste) = bootstrap(data)
    print("treino: "+ str(len(treino)) + "\n" + str(treino))
    print("teste: "+ str(len(teste)) + "\n" + str(teste))
    print("\nBOOTSTRAP 2:")
    (treino, teste) = bootstrap(data)
    print("treino: "+ str(len(treino)) + "\n" + str(treino))
    print("teste: "+ str(len(teste)) + "\n" + str(teste))

def testInduce(data):
    root = dt.DecisionTree()
    root.makeRootNode(data)
    root.induce(root.data, root.listOfAttr)
    root._print(0)
    return root

def testPlotting(tree):
	plot = pt.PlotTree(tree)
	plot.drawTree()

def test(data):
	tree = testInduce(data)
	#testPlotting(tree)
	testClassify(tree)
	testBootstrap(data)