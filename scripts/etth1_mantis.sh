if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
script_name=$(basename "$0" .sh)
model_name=PatchTST_REPA

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
pred_len=720

# Choose feature extractor: 'tivit' or 'mantis'
feature_extractor='mantis'

# Use projector for feature alignment (1: use, 0: original PatchTST)
use_projector=1

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 4 \
  --encoder_depth 4 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 20\
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --save_checkpoint 0 \
  --use_projector $use_projector \
  --feature_extractor $feature_extractor \
  --projector_dim 768 \
  --lambda_contrastive 0.5 \
  --tivit_pretrained ./open_clip/open_clip_model.safetensors \
  --mantis_pretrained ./Mantis \
  >logs/LongForecasting/${script_name}_${pred_len}.log