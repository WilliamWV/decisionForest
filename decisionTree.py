import pandas as pd
import math

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

	def _print(self, tabs):
		print("\t"*tabs, end='')
		print("Atributo " + str(self.attrIndex))
		if (self.attrType == CATEGORIC):
			print("\t"*(tabs+1), end='')
			print ("Tipo = CATEGORICO")
			print("\t"*(tabs+1), end='')
			print ("Valores = " + str(self.catVals))
		elif (self.attrType == NUMERIC):
			print("\t"*(tabs+1), end='')
			print ("Tipo = NUMERICO")
			print("\t"*(tabs+1), end='')
			print ("Ponto de corte = " + str(self.cutPoint))

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
	
	def __init__(self, predictedIndex=-1, answer=None):
		self.subtrees = {}						# é vazio se for nodo de resposta (ou seja, nodo folha)
		self.questionAttr = None				# é None se for nodo de resposta (ou seja, nodo folha)
		self.predictedIndex = predictedIndex	# índice do atributo a ser previsto
		self.answer = answer					# é None se for nodo de decisão (ou seja, nodo interno)
		self.nodeGain = 0
		self.data = None 						# é None, a não ser que seja nodo raiz
		self.listOfAttr = []					# é vazia, a não ser que seja nodo raiz
	
	def makeRootNode(self, data):
		self.data = data
		self.listOfAttr = self.listAttributes(self.data)

	def listAttributes(self, data):
		attributes = []
		for i in range(0, len(data.columns)):
			columnName = data.columns[self.predictedIndex] # nome da coluna a ser predita
			if columnName != data.columns[i]:
				values = data[data.columns[i]].unique()
				if(isinstance(values[0], str)):
					numericOrCategoric = CATEGORIC
				else:
					numericOrCategoric = NUMERIC
				attributes.append(Attr(numericOrCategoric, i, values))
		return attributes

	def addSubtree (self, newTree, decision):
		self.subtrees[decision] = newTree

	#instance deve ser uma instância dos dados de entrada que possui o mesmo 
	#formato quanto a ordem e os tipos dos atributos
	def classify(self, instance):
		pass

	def induce(self, D, L):
		columnName = D.columns[self.predictedIndex] # nome da coluna a ser predita
		column = D[columnName] # pega todos os dados da coluna
		columnValues = column.unique() # separa cada valor único da coluna
		
		if len(columnValues) == 1:
			self.answer = columnValues[0]
			return self
		
		mostFrequentValue = column.value_counts().idxmax() # pega o valor mais frequente da coluna
		if len(L) == 0:
			self.answer = mostFrequentValue
			return self
		
		bestAttr = self.selectAndRemoveAttr(L, D)
		self.questionAttr = bestAttr

		if bestAttr.attrType == CATEGORIC:
			columnName = D.columns[bestAttr.attrIndex]
			for i in range(0, len(bestAttr.catVals)):
				Dv = D.loc[ D[columnName] == bestAttr.catVals[i] ] # obtém todas as linhas cujo atributo bestAttr possua o valor atual (representado por bestAttr.catVals[i])
				if Dv.empty:
					columnName = D.columns[self.predictedIndex]
					column = D[columnName]
					mostFrequentValue = column.value_counts().idxmax()
					self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex, answer=mostFrequentValue), bestAttr.catVals[i] )
				else:
					self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex).induce(Dv, L), bestAttr.catVals[i] )
		
		return self
	
	def info(self, D):
		columnName = D.columns[self.predictedIndex]
		column = D[columnName]
		listOfOccurrences = column.value_counts().tolist()
		total = sum(listOfOccurrences)
		info_D = 0
		for i in listOfOccurrences:
			info_D = info_D + (i/total)*math.log2(i/total)
		info_D = (-1)*info_D
		return info_D
	
	def gain(self, L, D):
		info_D = self.info(D)
		index = self.predictedIndex
		attributesGain = []
		for i in L:
			columnName = D.columns[i.attrIndex]
			info_aD = 0
			for j in i.catVals:
				Dj = D.loc[ D[columnName] == j ]
				counter = len(Dj.index)
				info_aD = info_aD + (counter/len(D.index))*self.info(Dj)
			gain = info_D - info_aD
			attributesGain.append([gain, i])
		return attributesGain

	def selectAndRemoveAttr(self, L, D):
		attributesGain = self.gain(L, D)
		attributesGain.sort(key=lambda x: x[0], reverse=True)
		attr = attributesGain[0][1]
		self.nodeGain = attributesGain[0]
		for i in range(0, len(L)):
			if L[i].attrIndex == attr.attrIndex:
				del L[i]
				break
		return attr

	def _print(self, tabs):
		print("\t"*tabs, end='')
		print("Ganho do nodo: ", end="")
		print(self.nodeGain)
		if self.questionAttr is not None:	
			print("\t"*tabs, end='')
			print ("Nodo de decisão")
			print("\t"*tabs, end='')
			print ("Decisão dada por: ")		
			self.questionAttr._print(tabs)
		if self.answer is not None:
			print("\t"*tabs, end='')
			print("Nodo de resposta com resposta: ", end="")
			print(self.answer)
		if (len(self.subtrees) > 0):
			index = 0
			for key in self.subtrees:
				print("\t"*tabs, end='')
				if self.questionAttr.attrType == CATEGORIC:
					print("Subárvore " + str(index) + " (valor=", end="")
					print(key, end="")
					print("):")
				self.subtrees[key]._print(tabs+1)
				index += 1