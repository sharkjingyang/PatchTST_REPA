if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST_REPA_Fusion

root_path_name=./dataset/
data_path_name=ETTh1.csv
data_name=ETTh1

random_seed=2021
pred_len=720

# Choose feature extractor: 'tivit', 'mantis' or 'chronos'
feature_extractor='chronos'
contrastive_type='patch_wise'
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
  --e_layers 4 \
  --encoder_depth 2 \
  --d_layers 1 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3\
  --fc_dropout 0.3\
  --head_dropout 0\
  --patch_len 16\
  --stride 16\
  --padding_patch None\
  --head_type $head_type\
  --des 'Exp' \
  --train_epochs 20\
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --feature_extractor $feature_extractor \
  --contrastive_type $contrastive_type \
  --projector_dim 768 \
  --lambda_contrastive 0.5 \
  --tivit_pretrained ./open_clip/open_clip_model.safetensors \
  --mantis_pretrained ./Mantis \
  --chronos_pretrained ./Chronos2 \
  >logs/LongForecasting/${data_name}_${feature_extractor}_${contrastive_type}_${seq_len}_${pred_len}.log