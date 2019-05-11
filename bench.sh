#!/bin/bash

#dados
wine_data="wine/wine.data"
ionosphere_data="ionosphere/ionosphere.data"
wdbc_data="wdbc/wdbc.data"
DATASETS=$( echo $wine_data; echo $ionosphere_data; echo $wdbc_data; )

#informações do comando
PY="python3" #interpretador
MAIN="main.py"

#argumentos
FOLDS=$( echo "10";)
N_TREES=$( echo "1"; echo "2"; echo "5"; echo "10"; echo "20";)
BETAS=$( echo "0.5"; )
SCORES=$( echo "macro"; )
#prediction_index
declare -A pred_indexes=([$wine_data]="0" [$ionosphere_data]="-1" [$wdbc_data]="0") # declara um array associativo (HASH) # wdbc é 0 apesar de ser a segunda coluna pois ignora a primeira
#ignore -> id do wdbc
declare -A ignore=([$wine_data]="" [$ionosphere_data]="" [$wdbc_data]="-i id")

# Log:
# Todos os logs ficarão no arquivo indicado na variável LOG_FILE e estarão organizados da seguinte forma:
#   Dataset: $dataset; Fold size: $k; Num of trees: $t; Beta: $b; Score mode: $s
#   F-Score average = $average
#   F-score deviation = $deviation

LOG_FILE="logs.txt"
> $LOG_FILE # apaga logs anteriores


for dataset in $DATASETS; do
    for k in $FOLDS; do
        for t in $N_TREES; do
            for b in $BETAS; do
                for s in $SCORES; do
                    echo "Dataset: "$dataset"; Fold size: $k; Num of trees: $t; Beta: $b; Score mode: $s">>$LOG_FILE
                    $PY $MAIN -d $dataset -k $k -t $t -p ${pred_indexes[$dataset]} ${ignore[$dataset]} -b $b -s $s>>$LOG_FILE
                done
            done
        done
    done 
done