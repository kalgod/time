# export CUDA_VISIBLE_DEVICES=1

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/bandwidth/ \
  --data_path bandwidth.csv \
  --model_id bandwidth_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 5 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1 \
  --inverse