from main import parse
from main import bootstrap
from main import trainEnsemble
from main import vote
from main import getCategories

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
    root.makeRootNode(data, data)
    root.induce(root.data, root.listOfAttr)
    root._print(0)
    return root

def testPlotting(tree):
	plot = pt.PlotTree(tree)
	plot.drawTree()

def testEnsemble(data):
    print(" -------- Ensemble de 1 arvore --------")
    ensemble = trainEnsemble(data, 1)
    for i, e in enumerate(ensemble):
        print(" ## Arvore %d --------" % i)
        e._print(0)

    print(" -------- Ensemble de 3 arvores --------")
    ensemble = trainEnsemble(data, 3)
    for i, e in enumerate(ensemble):
        print(" ## Arvore %d --------" % i)
        e._print(0)

def testVote(data):
    inst1 = ["Nublado", "Quente", "Normal", "Falso"]  # Espera Sim
    inst2 = ["Nublado", "Quente", "Alta", "Falso"]    # Espera Sim

    categories, _ = getCategories(data, -1)

    print(" -------- Ensemble de 3 arvores --------")
    ensemble = trainEnsemble(data, 3)
    print("Classificacao da instancia 1: %s (era pra ser Sim)" % vote(ensemble, inst1, categories))
    print("Classificacao da instancia 2: %s (era pra ser Sim)" % vote(ensemble, inst2, categories))

def test(data):
	# print("######################## TESTE INDUCE ###########################")
	tree = testInduce(data)
	# print("######################## TESTE PLOTTING ###########################")
	testPlotting(tree)
	# print("######################## TESTE CLASSIFY ###########################")
	# testClassify(tree)
	# print("######################## TESTE BOOTSTRAP ###########################")
	# testBootstrap(data)
	# print("######################## TESTE ENSEMBLE ###########################")
	# testEnsemble(data)
	# print("######################## TESTE VOTE ###########################")
    # testVote(data)
