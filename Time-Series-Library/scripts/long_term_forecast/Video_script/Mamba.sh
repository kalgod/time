model_name=Mamba
pred_len=5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/video/ \
  --data_path gtscore.csv \
  --model_id weather_$pred_len'_'$pred_len \
  --model $model_name \
  --data video \
  --features S \
  --seq_len $pred_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 21 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 21 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10