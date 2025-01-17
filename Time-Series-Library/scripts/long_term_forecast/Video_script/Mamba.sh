model_name=DLinear
pred_len=8

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/video/ \
  --data_path gtscore_video.csv \
  --model_id weather_$pred_len'_'$pred_len \
  --model $model_name \
  --data video \
  --features S \
  --seq_len $pred_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 20