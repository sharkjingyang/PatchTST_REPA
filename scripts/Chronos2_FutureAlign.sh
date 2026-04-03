if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=336
model_name=PatchTST_future_align
device="cuda:0"

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=336
d_model=128
e_layers=3
n_heads=16
d_ff=256

# Joint distillation hyperparameters
lambda_t=0.5        # Phase 1 (warmup): teacher loss weight
lambda_t2=0.1       # Phase 2: teacher loss weight (smaller → slower drift)
lambda_a=0.5        # Phase 2: alignment loss weight
align_warmup=5      # epochs of teacher-only warmup before alignment starts

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
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0.0 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --lambda_t $lambda_t \
  --lambda_t2 $lambda_t2 \
  --lambda_a $lambda_a \
  --align_warmup_epochs $align_warmup \
  --device $device \
  >logs/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_dm${d_model}_el${e_layers}_lt${lambda_t}_la${lambda_a}.log
