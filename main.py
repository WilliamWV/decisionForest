import decisiontree as dt
import test
import pandas as pd
import random as rd

# Semente para geração dos números aleatórios, usada em fase de 
# desenvolvimento para permitir que os resultados possam ser repetidos
rd_seed = 135872916542

# Quantidade de árvores na floresta
ntree  = 10

# Floresta de árvores de decisão
forest = []


def parse(fileName):
    data = pd.read_csv(fileName, delimiter=';', header=0)
    dt.Data = data.values.tolist()
    return data

# Realização do bootstrap dos dados dividindo-os em conjunto de teste
# e de treinamento, a divisão é feita usando o seguinte valor que indica
# qual o tamanho do bootstrap comparado com o conjunto de treinamento
bootstrap_size = 0.9 # 90% do tamanho do conjunto original
# como o bootstrap realizado é com reposição, mesmo que o valor acima 
# seja igual a 1 (mesmo tamanho do conjunto original), o bootstrap de
# treinamento retornado, muito provavelmente será diferente do conjunto
# de dados originais, pois instâncias podem ser escolhidas múltiplas vezes
# a função bootstrap a seguir retorna um par, com duas listas, a primeira
# indica o conjunto de treinamento e a segunda indica o conjunto de teste
def bootstrap(data):
    train = pd.DataFrame()
    test = pd.DataFrame()
    choosed = []
    for _ in range(int(bootstrap_size * len(data))):
        index = rd.choice(range(len(data)))
        choosed.append(index)
        # Usando list slice aqui para que o pandas monte um DataFrame com a 
        # mesma estrutura, mas com apenas a linha selecionada de dados
        train = train.append(data[index:index+1]) 
    notChoosed = [i for i in range(len(data)) if i not in choosed]
    for i in notChoosed:
        test = test.append(data[i:i+1])
    
    return (train, test)
    

### Coisas para fazer quando estivermos com tudo pronto separadamente
### TODO: trocar o valor m da amostragem que atualmente é len(L) -> não possui efeitos
### TODO: chamar a funao bootstrap() para obter os dados de teste e de treinamento usados de fato
###       para treinar as árvores
### TODO: assinalar ao valor dt.Data o resultado da função bootstrap para o conjunto de treino
###       convertido para listas, ou seja: treino.values.tolist()

def main():
    fileName = "dadosBenchmark_validacaoAlgoritmoAD.csv"
    data = parse(fileName)
    test.test(data)


if __name__ == "__main__":
    rd.seed(rd_seed)
    main()