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

	# recebe uma nova instância e retorna um valor correspondente a sua classificação quanto
	# a esse atributo
	def decide (self, instance):
		if self.attrType == NUMERIC:
			if (instance[self.attrIndex] > self.cutPoint):
				return NUM_GREATER
			else:
				return NUM_SMALLER
		elif self.attrType == CATEGORIC:
			if instance[self.attrIndex] in self.catVals: 
				return instance[self.attrIndex]
			else:
				print ("ERROR: non-existing categoric value")
	

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
###  * As subárvores são representadas por um dicionário que associa         ###
###    decisões a subárvores, de modo que, a partir de uma decisão que       ###
###    corresponde a uma instância, se possa obter a próxima árvore para     ###
###    seguir a classificação                                                ###
###    ex:                                                                   ###
###         Nodo deve decide se atributo 1 é maior do que 7, se for maior    ###
###         (NUM_GREATER = -1) a classificação continua pela subárvore A     ###
###         se não (NUM_SMALLER = -2), a classificação continua pela         ###
###         subárvore B, nesse caso o membro "self.subtrees" seria:          ###
###         { -1 : A, -2 : B }                                               ###
###  * Responsável por:                                                      ###
###    - Classificar uma entrada conforme as decisões dos nodos              ###
###    - Induzir uma árvore de decisão                                       ###
################################################################################

class DecisionTree:
	
	def __init__(self, predictedIndex=-1, answer=None):
		self.subtrees = {}						# é vazio se for nodo de resposta (ou seja, nodo folha); a chave é a resposta para uma pergunta, o valor é a subárvore alcançada por tal resposta
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
		if (self.questionAttr == None):
			return self.answer
		else:
			return self.subtrees[self.questionAttr.decide(instance)].classify(instance)


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
		columnName = D.columns[bestAttr.attrIndex]

		if bestAttr.attrType == CATEGORIC:
			for i in range(0, len(bestAttr.catVals)):
				Dv = D.loc[ D[columnName] == bestAttr.catVals[i] ] # obtém todas as linhas cujo atributo bestAttr possua o valor atual (representado por bestAttr.catVals[i])
				if Dv.empty:
					columnName = D.columns[self.predictedIndex]
					column = D[columnName]
					mostFrequentValue = column.value_counts().idxmax()
					self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex, answer=mostFrequentValue), bestAttr.catVals[i] )
				else:
					self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex).induce(Dv, L), bestAttr.catVals[i] )
		else:
			Dv = D.loc[ D[columnName] <= bestAttr.cutPoint ]
			if Dv.empty:
				columnName = D.columns[self.predictedIndex]
				column = D[columnName]
				mostFrequentValue = column.value_counts().idxmax()
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex, answer=mostFrequentValue), NUM_SMALLER )
			else:
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex).induce(Dv, L), NUM_SMALLER )
			Dv = D.loc[ D[columnName] > bestAttr.cutPoint ]
			if Dv.empty:
				columnName = D.columns[self.predictedIndex]
				column = D[columnName]
				mostFrequentValue = column.value_counts().idxmax()
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex, answer=mostFrequentValue), NUM_GREATER )
			else:
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex).induce(Dv, L), NUM_GREATER )
		
		return self
	
	def info(self, D):
		columnName = D.columns[self.predictedIndex]
		column = D[columnName]
		listOfOccurrences = column.value_counts().tolist() # lista com o número de ocorrências de cada valor do atributo a ser predito
		rows = sum(listOfOccurrences)
		infoD = 0
		for i in listOfOccurrences:
			infoD = infoD + (i/rows)*math.log2(i/rows)
		infoD = (-1)*infoD
		return infoD
	
	def gain(self, L, D):
		infoD = self.info(D)
		index = self.predictedIndex
		attributesGain = []
		for i in L:
			columnName = D.columns[i.attrIndex] # nome do atributo cujo ganho está sendo calculado
			infoAD = 0
			if i.attrType == CATEGORIC:
				for j in i.catVals:
					Dj = D.loc[ D[columnName] == j ] # obtém as linhas cujo valor do atributo em questão seja igual a J
					counter = len(Dj.index) # número de linhas em Dj
					infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
			else:
				Dj = D.loc[ D[columnName] <= i.cutPoint ]
				counter = len(Dj.index)
				infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
				Dj = D.loc[ D[columnName] > i.cutPoint ]
				counter = len(Dj.index)
				infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
			gain = infoD - infoAD
			attributesGain.append([gain, i])
		return attributesGain

	def selectAndRemoveAttr(self, L, D):
		attributesGain = self.gain(L, D)
		attributesGain.sort(key=lambda x: x[0], reverse=True)
		attr = attributesGain[0][1]
		self.nodeGain = attributesGain[0]
		for i in range(0, len(L)):
			if L[i] == attr:
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
				print("Subárvore " + str(index) + " (valor: ", end="")
				if (self.questionAttr != None and self.questionAttr.attrType == NUMERIC):
					if (key == NUM_GREATER):
						print ("> " + str(self.questionAttr.cutPoint), end="")
					else:
						print ("<= " + str(self.questionAttr.cutPoint), end="")
				else:
					print(key, end="")
				print("):")
				self.subtrees[key]._print(tabs+1)
				index += 1