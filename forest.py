# -*- coding: utf-8 -*-
'''
	Implementação do algoritmo de aprendizado supervisionado de florestas 
	aleatórias para classificação de dados construindo um ensemble de árvore 
	de decisão.	


	Alunos:
		* Aline Weber
		* Felipe Zorzo
		* William Wilbert
'''


##### TIPOS DOS ATRIBUTOS ACEITOS  #####
CATEGORIC  =  0                      ###
NUMERIC    =  1                      ###
########################################

##### RESULTADOS DE UMA DECISÃO DE UM NODO #####
NUM_GREATER = -1 # Valor numérico maior que o ponto de corte
NUM_SMALLER = -2 # Valor numérico menor que o ponto de corte
#USAR VALORES DIFERENTES PARA ATRIBUTOS CATEGÓRICOS
################################################

# Dados de entrada
Data = [[]]

# Quantidade de árvores na floresta
ntree  = 10

#Floresta de árvores de decisão
forest = []

################################################################################
### class Attr:                                                              ###
###  * Usada para representar os atributos das instâncias de entrada sendo   ### 
###    responsável por:                                                      ###
###     - determinação de pontos de corte em atributos numéricos             ###
###     - retorno de todos os valores possíveis de atributos categóricos     ###
###  * Fará parte da estrutura básica de um aárvore de decisão pois          ###
###    representa a decisão feita em um nodo                                 ###
################################################################################
class Attr:

	def __init__(self, attrType, attrIndex, values):
		self.attrType = attrType
		self.attrIndex = attrIndex
		
		if (attrType == CATEGORIC):
			self.catVals = values
		elif (attrType == NUMERIC):
			self.cutPoint = self._calcCutPoint()					

	def getCutPoint(self): 
		if self.attrType != NUMERIC:
			return None
		else:
			return self.cutPoint


	def getCatVals(self): 
		if self.attrType != CATEGORIC:
			return None
		else:
			return self.catVals


	def _calcCutPoint(self):
		#Usa média dos valores
		return sum([Data[i][self.attrIndex] for i in range(len(Data))]) / len(Data)
	def _print(self):
		print("Atributo " + str(self.attrIndex))
		if (self.attrType == CATEGORIC):
			print ("    Tipo = CATEGORICO")
			print ("    Valores = " + str(self.catVals))
		elif (self.attrType == NUMERIC):
			print ("    Tipo = NUMERICO")
			print ("    Ponto de corte = " + str(self.cutPoint))
		print("")
		

################################################################################
### class DecisionTree:                                                      ###
###  * Usada para representar uma árvore de decisão, cada nodo da árvore     ### 
###    possui a representação da decisão que é feita nesse nodo bem como sua ###
###    lista de subárvores                                                   ###
###  * Cada subárvore é representada por uma tupla onde o primeiro elemento  ###
###    é a ssubárvore em si do tipo DecisionTree e o segundo é um valor      ###
###    numérico que indica qual o resultado da decisão que deve ocorrer      ###
###    nesse nodo para que a classificação continue na subárvore da tupla,   ###
###    ex:                                                                   ###
###         Nodo deve decidir se atributo 1 é maior do que 7, se for maior   ###
###         (NUM_GREATER = -1) a classificação continua pela subárvore A     ###
###         se não (NUM_SMALLER = -2), a classificação continua pela         ###
###         subárvore B, nesse caso o membro "self.subtrees" seria:          ###
###         [(A, -1), (B, -2)]                                               ###
###  * Responsável por:                                                      ###
###    - Classificar uma entrada conforme as decisões dos nodos              ###
################################################################################

class DecisionTree:
	
	def __init__(self, question):
		self.subtrees = []
		self.questionAttr = question
	
	# Antes de inserir uma nova subárvore verifica se a decisão correspondente 
	# já não foi associada a outra subárvore nesse mesmo nodo
	def addSubtree (self, newTree, decisionResult):
		if (decisionResult in [self.subtrees[i][1] for i in range(len(self.subtrees))]):
			print("Failed to insert subtree once the corresponding result already exists on this node")
		else:
			self.subtrees.append((newTree, decisionResult))
	

	#instance deve ser uma instância dos dados de entrada que possui o mesmo 
	#formato quanto a ordem e os tipos dos atributos
	def classify(self, instance):
		pass

	def _print(self):
		print ("Nodo de decisão")
		print ("Decisão dada por: ")
		self.questionAttr._print()
		if (len(self.subtrees) > 0):		
			print ("Subárvores:")
			for i in self.subtrees:
				print ("Decisão correspondente = " + str(i[1]))
				i[0]._print()
		

################################################################################
### TESTES - APAGAR QUANDO O ALGORITMO DE INDUÇÃO FOR IMPLEMENTADO          ####
################################################################################
Data = [[1.1, 0, 3], [3.7, 1, 6], [-1.6, 0, 9]]

attr1 = Attr(NUMERIC, 0, None)
attr2 = Attr(CATEGORIC, 1, [0, 1])
attr3 = Attr(NUMERIC, 2, None)

attr1._print()
attr2._print()
attr3._print()

root = DecisionTree(attr1)
root.addSubtree(DecisionTree(attr2), -1)
root.addSubtree(DecisionTree(attr3), -2)

root._print()


