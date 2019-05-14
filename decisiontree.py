import pandas as pd
import math
import random as rd
import numpy as np

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

	def __init__(self, attrType, attrIndex, values, attrName, data, predictedIndex):
		self.attrType = attrType
		self.attrIndex = attrIndex
		self.attrName = attrName
		
		if (attrType == CATEGORIC):
			self.catVals = values
		elif (attrType == NUMERIC):
			self.cutPoint = self._calcCutPoint(data, predictedIndex)	

	def isNumeric(self):
		if self.attrType == NUMERIC:
			return True
		else:
			return False
	
	def isCategoric(self):
		return not(self.isNumeric())

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

	def _calcCutPoint(self, data, predictionIndex):

		############# OPÇÕES DE PONTO DE CORTE #############
		######### Medidas estatísticas simples: ############
		# Média: Possuiu os melhores resultados dentre as medidas simples
		# return [np.mean(data[data.columns[self.attrIndex]].tolist())]
		# Mediana: 
		# return [np.median(data[data.columns[self.attrIndex]].tolist())]
		# Moda: problema com a moda: para usar essas bibliotecas do numpy precisa que os valores sejam positivos
		# os resultados experimentais da moda foram os piores dos três testados
		# return [np.bincount(data[data.columns[self.attrIndex]].tolist()).argmax()]
		## Obtenção de valores que melhor dividem as classes ##
		# Como nos slides: vantagem: ponto de corte melhor, desvantagem: potencialmente muito custo de execução
		# 1) Ordenar os valores do atributos
		possibleCuts = []
		s_vals = data.sort_values(by=[data.columns[self.attrIndex]])
		for i in range(1, len(s_vals)):
			preds = s_vals[data.columns[predictionIndex]].tolist()
			vals = s_vals[data.columns[self.attrIndex]].tolist()
			if preds[i] != preds[i-1]:
				avg = (vals[i] + vals[i-1])/2
				possibleCuts.append(avg)
		return possibleCuts

		# Ideia 1: Objetivo: ser mais rápido que a implementação dos slides mas tentar aproximar a precisão
		# Útil quando o número de classes é muito menor que o número de instâncias e a distribuição dos valores
		# dos atributos segue uma distribuição normal (se não seguir a separação por uma média simples pode ser
		# imprecisa)
		# Método:
		#  1) Calcular a média dos valores desse atributo de cada uma das classes
		#  2) Formar uma lista com todas as combinações duas a duas desass médias (realizar a média de cada par)
		#  3) Determinar qual dos valores de 2 possui o maior ganho de informação
		# Ideia 2: Objetivo: ser mais rápida que a ideia um gerando apenas um valor 
		# Útil quando a quantidade de classes se aproxima da quantidade de instâncias (acho que isso deve ser bem raro)
		#  1)  

	# recebe uma nova instância e retorna um valor correspondente a sua classificação quanto
	# a esse atributo
	def decide (self, instance, predictedIndex, cutPoint):
		if self.attrType == NUMERIC:
			if (instance[self.attrIndex-(1+predictedIndex)] > cutPoint):
				return NUM_GREATER
			else:
				return NUM_SMALLER
		elif self.attrType == CATEGORIC:
			if instance[self.attrIndex-(1+predictedIndex)] in self.catVals: 
				return instance[self.attrIndex-(1+predictedIndex)]
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
		self.cutPoint = math.inf				# soh sera diferente de infinito para nodos de decisao com questionAttr numerico
		self.predictedIndex = predictedIndex	# índice do atributo a ser previsto
		self.answer = answer					# é None se for nodo de decisão (ou seja, nodo interno)
		self.nodeGain = 0
		self.data = None 						# é None, a não ser que seja nodo raiz
		self.listOfAttr = []					# é vazia, a não ser que seja nodo raiz
	
	def makeRootNode(self, data, all_data):
		self.data = data
		self.listOfAttr = self.listAttributes(all_data)

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
				attributes.append(Attr(numericOrCategoric, i, values, data.columns[i], self.data, self.predictedIndex))
		return attributes

	def addSubtree (self, newTree, decision):
		self.subtrees[decision] = newTree

	#instance deve ser uma instância dos dados de entrada que possui o mesmo 
	#formato quanto a ordem e os tipos dos atributos
	def classify(self, instance):
		if (self.questionAttr == None):
			return self.answer
		else:
			return self.subtrees[self.questionAttr.decide(instance, self.predictedIndex, self.cutPoint)].classify(instance)


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
		bestAttr, cutPoint = self.selectAndRemoveAttr(L, D)
		self.questionAttr = bestAttr
		self.cutPoint = cutPoint
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
			Dv = D.loc[ D[columnName] <= cutPoint ]
			if Dv.empty:
				columnNamePred = D.columns[self.predictedIndex]
				column = D[columnNamePred]
				mostFrequentValue = column.value_counts().idxmax()
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex, answer=mostFrequentValue), NUM_SMALLER )
			else:
				self.addSubtree( DecisionTree(predictedIndex=self.predictedIndex).induce(Dv, L), NUM_SMALLER )
			Dv = D.loc[ D[columnName] > cutPoint ]
			if Dv.empty:
				columnNamePred = D.columns[self.predictedIndex]
				column = D[columnNamePred]
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
	
	# A amostragem dos atributos deve ocorrer a cada nodo da árvore, portanto, na mesma árvore, cada
	# nodo deve considerar um subconjunto dos atributos que potencialmente será diferente dos demais
	# subconjuntos dos outros nodos da mesma subárvore. Portanto a amostragem deve ser feita de forma
	# independente dentro de cada nodo, a opção escolhida nesse caso é implementar a amostragem dos
	# atributos dentro da função "gain" abaixo, pois ela calcula quanto será o nodo de cada 
	
	def gain(self, L, D):
		infoD = self.info(D)
		index = self.predictedIndex
		attributesGain = []
		#m = len(L) # Quantidade de atributos na amostragem
		m = int(math.ceil(math.sqrt(len(L)))) # usando m como raiz quadrada -> alternativa sugerida nos slides
		attributesSample = rd.sample(L, m)
		cut = [math.inf, math.inf]
		for i in attributesSample:
			columnName = D.columns[i.attrIndex] # nome do atributo cujo ganho está sendo calculado
			infoAD = 0
			if i.attrType == CATEGORIC:
				for j in i.catVals:
					Dj = D.loc[ D[columnName] == j ] # obtém as linhas cujo valor do atributo em questão seja igual a J
					counter = len(Dj.index) # número de linhas em Dj
					infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
			else:
				for j in i.cutPoint:
					infoAD = 0
					Dj = D.loc[ D[columnName] <= j ]
					counter = len(Dj.index)
					infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
					Dj = D.loc[ D[columnName] > j ]
					counter = len(Dj.index)
					infoAD = infoAD + (counter/len(D.index))*self.info(Dj)
					if infoAD < cut[0]: # menor infoAD ira gerar ganho maior
						cut[0] = infoAD
						cut[1] = j
			if i.attrType == CATEGORIC:
				gain = infoD - infoAD
			else:
				gain = infoD - cut[0]				
			attributesGain.append([gain, i, cut[1]])
		return attributesGain

	def selectAndRemoveAttr(self, L, D):
		attributesGain = self.gain(L, D)
		attributesGain.sort(key=lambda x: x[0], reverse=True)
		attr = attributesGain[0][1]
		cutPoint = attributesGain[0][2]
		self.nodeGain = attributesGain[0]
		for i in range(0, len(L)):
			if L[i] == attr:
				del L[i]
				break
		return attr, cutPoint

	def isAnswerNode(self):
		if self.answer is not None:
			return True
		else:
			return False
	
	def isQuestionNode(self):
		return not(self.isAnswerNode())

	def _print(self, tabs):
		print("\t"*tabs, end='')
		print("Ganho do nodo: ", end="")
		print(self.nodeGain)
		if self.isQuestionNode():	
			print("\t"*tabs, end='')
			print ("Nodo de decisão")
			print("\t"*tabs, end='')
			print ("Decisão dada por: ")		
			self.questionAttr._print(tabs)
		if self.isAnswerNode():
			print("\t"*tabs, end='')
			print("Nodo de resposta com resposta: ", end="")
			print(self.answer)
		if (len(self.subtrees) > 0):
			index = 0
			for key in self.subtrees:
				print("\t"*tabs, end='')
				print("Subárvore " + str(index) + " (valor: ", end="")
				if (self.isQuestionNode() and self.questionAttr.attrType == NUMERIC):
					if (key == NUM_GREATER):
						print ("> " + str(self.cutPoint), end="")
					else:
						print ("<= " + str(self.cutPoint), end="")
				else:
					print(key, end="")
				print("):")
				self.subtrees[key]._print(tabs+1)
				index += 1