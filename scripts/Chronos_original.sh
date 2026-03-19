if [ ! -d "./logs" ]; then mkdir ./logs; fi

seq_len=336
model_name=Chronos_original
root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1
random_seed=2021
pred_len=96

python -u test_Chronos2_direct.py \
  --random_seed $random_seed \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${data_name}_${seq_len}_${pred_len} \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --batch_size 128 \
  --chronos_pretrained ./Chronos2 \
  >logs/${data_name}_${seq_len}_${pred_len}_${model_name}.log
