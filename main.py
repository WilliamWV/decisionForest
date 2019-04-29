import parseData
import decisionTree as dt

# Quantidade de árvores na floresta
ntree  = 10

#Floresta de árvores de decisão
forest = []

def main():
    fileName = "dadosBenchmark_validacaoAlgoritmoAD.csv"
    parseData.parse(fileName)
    root = dt.DecisionTree()
    root.induce(dt.dataPanda, dt.listOfAttr)
    root._print(0)

    '''
    attr1 = dt.Attr(dt.NUMERIC, 0, None)
    attr2 = dt.Attr(dt.CATEGORIC, 1, [0, 1])
    attr3 = dt.Attr(dt.NUMERIC, 2, None)

    attr1._print()
    attr2._print()
    attr3._print()

    root = dt.DecisionTree(attr1)
    root.addSubtree(dt.DecisionTree(attr2), -1)
    root.addSubtree(dt.DecisionTree(attr3), -2)

    root._print()
    '''

if __name__ == "__main__":
    main()