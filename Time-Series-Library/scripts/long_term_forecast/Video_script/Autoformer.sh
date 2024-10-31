# export CUDA_VISIBLE_DEVICES=1

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/video/ \
  --data_path gtscore.csv \
  --model_id bandwidth_16_16 \
  --model $model_name \
  --data video \
  --features S \
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
  --train_epochs 10 \
  --inverse