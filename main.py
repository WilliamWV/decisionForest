import decisiontree as dt
import test
import pandas as pd
import random as rd
import numpy as np
import operator 
import argparse
import time

# Semente para geração dos números aleatórios, usada em fase de 
# desenvolvimento para permitir que os resultados possam ser repetidos
rd_seed = 6041


# Floresta de árvores de decisão
forest = []


def getCategories(data, predictionIndex):
    columnName = data.columns[predictionIndex] # nome da coluna a ser predita
    column = data[columnName]                  # pega todos os dados da coluna
    columnValues = column.unique()             # separa cada valor único da coluna
    numberOfInstances = dict(column.value_counts())

    return columnValues, numberOfInstances


def parse(fileName, ignore):
    data = pd.read_csv(fileName, delimiter=',', header=0)
    
    if ignore:
        data = data.drop(columns=[ignore])
    
    dt.Data = data.values.tolist()
    return data

# Realização do bootstrap dos dados dividindo-os em conjunto de teste
# e de treinamento.
# Como o bootstrap realizado é com reposição, mesmo que o valor acima 
# seja igual a 1 (mesmo tamanho do conjunto original), o bootstrap de
# treinamento retornado, muito provavelmente será diferente do conjunto
# de dados originais, pois instâncias podem ser escolhidas múltiplas vezes
# a função bootstrap a seguir retorna um par, com duas listas, a primeira
# indica o conjunto de treinamento e a segunda indica o conjunto de teste
def bootstrap(data):
    train = pd.DataFrame()
    test = pd.DataFrame()
    choosed = []
    for _ in range(len(data)):
        index = rd.choice(range(len(data)))
        choosed.append(index)
        # Usando list slice aqui para que o pandas monte um DataFrame com a 
        # mesma estrutura, mas com apenas a linha selecionada de dados
        train = train.append(data[index:index+1]) 
    notChoosed = [i for i in range(len(data)) if i not in choosed]
    for i in notChoosed:
        test = test.append(data[i:i+1])
    
    return (train, test)



def trainEnsemble(data, numTrees,predictionIndex):
    ensemble = []
    for tree_index in range(numTrees):
        train, test = bootstrap(data)
        # print("treino: "+ str(len(train)) + "\n" + str(train))

        root = dt.DecisionTree(predictedIndex=predictionIndex)
        root.makeRootNode(train, data)
        root.induce(root.data, root.listOfAttr)

        ensemble += [root]

    return ensemble


def vote(ensemble, instance, categories):
    answers = {}
    for category in categories:
        answers[category] = 0

    for tree_index, tree in enumerate(ensemble):
        ans = tree.classify(instance)
        answers[ans] += 1

    answers = sorted(answers.items(), key=operator.itemgetter(1), reverse=True)

    for category, votes in answers:
        return category


def cross_validation(data, predictionIndex, k, numTrees, beta, score_mode):
    categories, numberOfInstances = getCategories(data, predictionIndex)

    predictionColumnName = data.columns[predictionIndex]
    data_split_per_category = {}

    for category in categories:
        data_split_per_category[category] = data[data[predictionColumnName]==category]

    k_folds = [pd.DataFrame()] * k
    fscore = []

    # dividindo os folds estratificados
    for i in range(k):
        for category in categories:
            num_sample = numberOfInstances[category]//k
            sample = data_split_per_category[category].sample(n=num_sample)
            k_folds[i] = k_folds[i].append(sample)
            data_split_per_category[category] = data_split_per_category[category].drop(sample.index)


    # adicionando as instâncias que sobraram por categoria uma em cada fold (pra não ficar muito desparelha a quantidade total de instancias)
    instances_rest = pd.DataFrame()
    for category, data in data_split_per_category.items():
        instances_rest = instances_rest.append(data)

    fold_index = 0
    for i in range(len(instances_rest.index)):
        fold_index %= k
        k_folds[fold_index] = k_folds[fold_index].append(instances_rest.iloc[[i]])
        fold_index += 1


    # rodando cross-validation de fato
    for test_fold_index, testing_data in enumerate(k_folds):
        #agrupando folds restantes em um dataframe só
        training_data = pd.DataFrame()
        for fold_index, fold in enumerate(k_folds):
            if fold_index != test_fold_index:
                training_data = training_data.append(fold)
        
        ensemble = trainEnsemble(training_data, numTrees, predictionIndex)

        # classifica cada instancia usando o ensemble que acabou de aprender
        results = []
        for index, instance in testing_data.iterrows():
            instance = instance.tolist()
            if predictionIndex == -1:
                instance_classification = vote(ensemble, instance[:-1], categories)
            else:
                instance_classification = vote(ensemble, instance[1:], categories)
            results += [[instance[predictionIndex], instance_classification]]
        
        # print(results)
        fscore += [Fmeasure(results, categories, beta, score_mode)]

    
    print("F-Score average = %f" % np.mean(fscore))
    print("F-Score deviation = %f" % np.std(fscore))


def Fmeasure(results, categories, beta, score_mode):
    VPac = VNac = FPac = FNac = 0 # valores acumulados para todas as classes, a ser utilizado no caso de score_mode = micro média 
    all_precision = []
    all_recall = [] # resultados de precisão e recall para cada classe, a ser utilizado no caso de score_mode = macro média
    # Tratamento multiclasse realizado independentemente da quantidade de classes
    for category in categories:
        VP = VN = FP = FN = 0
        for r in results:
            if r[0] == category and r[1] == category:
                VP += 1
            elif r[0] == category and r[1] != category:
                FN +=1
            elif r[0] != category and r[1] == category:
                FP +=1
            else:
                VN += 1
        
        # print("VP = %d" % VP)
        # print("VN = %d" % VN)
        # print("FP = %d" % FP)
        # print("FN = %d" % FN)
        
        if score_mode == 'macro':
            # Evitar divisão por zero
            if VP + FP == 0:
                precision = 0
            else:
                precision = VP / (VP + FP)
            if VP + FN == 0:
                recall = 0
            else:
                recall = VP / (VP + FN)
            
            all_precision += [precision]
            all_recall += [recall]
        else:
            VPac += VP
            VNac += VN
            FPac += FP
            FNac += FN

    if score_mode == 'micro':
        precision = VPac / (VPac + FPac)
        recall = VPac / (VPac + FNac)
    else: # macro
        precision = np.mean(all_precision)
        recall = np.mean(all_recall)

    fscore = (1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return fscore


def main():
    parser = argparse.ArgumentParser(description="Random forests")


    parser.add_argument("-d", "--dataset", required=True, type=str, 
                        help="filename of the dataset")

    parser.add_argument("-k", "--folds_number", required=True, type=int,
                        help="the number of folders to divide the dataset in cross-validation")

    parser.add_argument('-t', '--trees_number', required=True, type=int,
                        help='number of trees in the ensemble')

    parser.add_argument('-p', '--prediction_index', required=True, type=int,
                        help="the index of the column with the classification info")

    parser.add_argument('-i', '--ignore', type=str,
                        help="the name of a column to ignore e.g. ID")

    parser.add_argument('-b', '--beta', type=float, default=1,
                        help="beta value of fmeasure")

    parser.add_argument('-s', '--score_mode', type=str, default="macro",
                        help="type of mean (macro or micro) to use on the fmeasure calc in case the data is multiclass")

    args = parser.parse_args()

    # fileName = "dadosBenchmark_validacaoAlgoritmoAD.csv"
    # fileName = "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    data = parse(args.dataset, args.ignore)
    # print(data)
    # test.test(data)
    start = time. time()
    cross_validation(data, args.prediction_index, args.folds_number, args.trees_number, args.beta, args.score_mode)
    end = time. time()
    print("Execution time = %f" % (end - start))


if __name__ == "__main__":
    rd.seed(rd_seed)
    np.random.seed(rd_seed)
    main()

