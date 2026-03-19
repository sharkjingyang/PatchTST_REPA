if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=336
model_name=Chronos2_head

root_path_name=./dataset/
data_path_name=weather.csv
data_name=weather

random_seed=2021
pred_len=96

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
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  >logs/${data_name}_${seq_len}_${pred_len}_${model_name}.log
