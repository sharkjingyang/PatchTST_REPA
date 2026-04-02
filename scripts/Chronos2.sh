if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=336
model_name=PatchTST_REPA
device="cuda:0"

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=720
d_model=16
d_ff=128
e_layers=3
encoder_depth=3
lambda_contrastive=0.5
lr=0.0001

# Choose feature extractor: 'tivit', 'mantis' or 'chronos'
feature_extractor='chronos'
contrastive=1
contrastive_type='patch_wise_cos'
head_type='flatten'

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
  --d_layers 0 \
  --n_heads 4 \
  --d_model $d_model \
  --d_ff $d_ff \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 16\
  --padding_patch None\
  --contrastive $contrastive\
  --head_type $head_type\
  --des 'Exp' \
  --train_epochs 20\
  --itr 1 --batch_size 128 --learning_rate $lr \
  --feature_extractor $feature_extractor \
  --contrastive_type $contrastive_type \
  --projector_dim 768 \
  --lambda_contrastive $lambda_contrastive \
  --device $device \
  >logs/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_dm${d_model}_el${e_layers}_${feature_extractor}_ct${contrastive}_lc${lambda_contrastive}_${contrastive_type}_${head_type}.log