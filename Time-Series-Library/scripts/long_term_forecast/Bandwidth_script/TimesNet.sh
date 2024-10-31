# export CUDA_VISIBLE_DEVICES=5

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/bandwidth/ \
  --data_path bandwidth_100000.csv \
  --model_id bandwidth_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 512 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1 \
  --inverse