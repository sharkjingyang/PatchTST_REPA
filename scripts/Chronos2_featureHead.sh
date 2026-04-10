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
pred_len=96
d_model=128

chronos_embed_type=past   # past | predict | future
#   past:    embed(x_past) past tokens → Flatten_Head  (verify teacher capacity with proj_down)
#   predict: encode(x_past) future tokens → PatchwiseHead  (≈ Chronos2 zero-shot quality)
#   future:  embed(x_future) teacher-forcing → head  (upper-bound experiment)

proj_down=1               # 0 | 1  (1 = Linear(768→d_model) before head)

head_type=flatten         # flatten | patch_wise
#   flatten:    Flatten_Head — works for all embed_types
#   patch_wise: PatchwiseHead — only meaningful for predict / future

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
  --d_model $d_model \
  --patch_len 16 \
  --chronos_embed_type $chronos_embed_type \
  --proj_down $proj_down \
  --head_type $head_type \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  --device $device \
  >logs/${model_name}_${data_name}_sl${seq_len}_pl${pred_len}_et${chronos_embed_type}_pd${proj_down}_${head_type}.log
