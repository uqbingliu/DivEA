#!/bin/bash

script_dir=$(dirname "$PWD/${0}")
. $script_dir/env_settings.sh


gpu_ids="0,1,2"


kgids="fb,dbp"
data_name="2m/fb_dbp/"
subtask_num=200

ea_model="gcn-align"
subtask_size=39874
ctx_g1_percent=0.4

alpha=0.9
beta=1.0
gamma=2.0
topK=10
max_iteration=5

ctx_builder=v1
train_percent=0.20
max_train_epoch=400
layer_num=2
seed=1011
ctx_g2_conn_percent=0.0


task="overperf_${ea_model}_${seed}/N${subtask_num}_S${subtask_size}_ctxb${ctx_builder}_g1Ctx${ctx_g1_percent}_g2CtxConn${ctx_g2_conn_percent}_${train_percent}_alpha${alpha}_beta${beta}_gamma${gamma}_topK${topK}"


. $script_dir/fn_settings.sh


# task cmds
cp -r ${dataset_root_dir}/../${data_name}/* ${data_dir}/


export CUDA_VISIBLE_DEVICES=$gpu_ids

params="--data_dir=${data_dir} --kgids=${kgids} --train_percent=${train_percent} --divide --seed=${seed}"
python ${proj_dir}/divea/run_prepare_data.py ${params}

params="--data_dir=${data_dir} --kgids=${kgids} --output_dir=${output_dir}
--subtask_num=${subtask_num} --subtask_size=${subtask_size} --ctx_g1_percent=${ctx_g1_percent} --ctx_g2_conn_percent=${ctx_g2_conn_percent}
--ctx_builder=${ctx_builder}
--ea_model=${ea_model}
--alpha=${alpha} --beta=${beta} --gamma=${gamma} --topK=${topK} --max_iteration=${max_iteration}
--max_train_epoch=${max_train_epoch} --layer_num=${layer_num}
--gpu_ids=${gpu_ids} --py_exe_fn=${py_exe_full_fn}
--seed=${seed}"
echo $params
python ${proj_dir}/divea/run_divea.py ${params}




