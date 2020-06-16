#!/bin/bash
DATA="${1-DD}"
fold=${2-1}  # 0 for 10-fold
GPU=${3-0}

seed=1

CONFIG=configs/${DATA}
if [ ! -f "$CONFIG" ]; then
    echo "No config file for ${DATA} in configs folder"
    exit 128
fi
source configs/${DATA}

FOLDER=results
FILE=${FOLDER}/${DATA}.txt
if [ ! -d "$FOLDER" ]; then
    mkdir $FOLDER
fi

run(){
    CUDA_VISIBLE_DEVICES=${GPU} python3 src/main.py \
        -seed $seed -data $DATA -fold $1 -num_epochs $num_epochs \
        -batch $batch_size -lr $learning_rate -deg_as_tag $deg_as_tag \
        -l_num $layer_num -h_dim $hidden_dim -l_dim $layer_dim \
        -drop_n $drop_network -drop_c $drop_classifier \
        -act_n $activation_network -act_c $activation_classifier \
        -ks $pool_rates_layers -acc_file $FILE
}

if [ ${fold} == 0 ]; then
    if [ -f "$FILE" ]; then
        rm $FILE
    fi
    echo "Running 10-fold cross validation"
    start=`date +%s`
    run $fold
    stop=`date +%s`
    echo "End of cross-validation using $[stop - start] seconds"
    echo "The accuracy results for ${DATA} are as follows:"
    cat $FILE
    echo "Mean and sstdev are:"
    cat $FILE | datamash mean 2 sstdev 2
else
    run $fold
    echo "The accuracy result for ${DATA} fold ${fold} is:"
    tail -1 $FILE
fi
