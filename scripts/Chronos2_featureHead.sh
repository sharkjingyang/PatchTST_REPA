if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=336
model_name=Chronos2_head
device="cuda:0"

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=720
use_future_patch=0

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${data_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --patch_len 16 \
  --use_future_patch $use_future_patch \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --device $device \
  >logs/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_ufp${use_future_patch}.log
