import decisionTree as dt
import pandas as pd

def listAttributes(data):
    attributes = []
    for i in range(0, len(data.columns)-1):
        values = data[data.columns[i]].unique()
        if(isinstance(values[0], str)):
            #print("Categorico: " + values[0])
            numericOrCategoric = dt.CATEGORIC
        else:
            #print("Numerico: " + str(values[0]))
            numericOrCategoric = dt.NUMERIC
        attributes.append(dt.Attr(numericOrCategoric, i, values))
    return attributes
            
def parse(fileName):
    data = pd.read_csv(fileName, delimiter=';', header=0)
    dt.dataPanda = data
    dt.Data = data.values.tolist()
    dt.listOfAttr = listAttributes(data)