if [ ! -d "./logs" ]; then mkdir ./logs; fi

seq_len=336
model_name=PatchTST_FM_zeroshot
device="cuda:0"
root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1
random_seed=2021
pred_len=96

python -u models/PatchTST_FM_zeroshot.py \
  --random_seed $random_seed \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${data_name}_${seq_len}_${pred_len} \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --batch_size 128 \
  --device $device \
  >logs/${data_name}_${seq_len}_${pred_len}_${model_name}.log
