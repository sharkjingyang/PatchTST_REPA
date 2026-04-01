if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=PatchTST_REPA_Fusion
device="cuda:0"

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=96
d_model=128
d_ff=256
e_layers=3
encoder_depth=2

# Fusion / alignment settings
patch_fusion_type=none        # none: patch_len auto = seq_len // (pred_len//16), align future tokens
feature_extractor=chronos
contrastive=1
contrastive_type=patch_wise
head_type=flatten
lambda_contrastive=0.1

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
  --encoder_depth $encoder_depth \
  --d_layers 1 \
  --n_heads 16 \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0 \
  --contrastive $contrastive \
  --head_type $head_type \
  --patch_fusion_type $patch_fusion_type \
  --feature_extractor $feature_extractor \
  --contrastive_type $contrastive_type \
  --lambda_contrastive $lambda_contrastive \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --device $device \
  >logs/LongForecasting/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_dm${d_model}_el${e_layers}_${patch_fusion_type}_${feature_extractor}_lc${lambda_contrastive}.log
