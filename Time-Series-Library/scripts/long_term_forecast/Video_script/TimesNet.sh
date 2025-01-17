# export CUDA_VISIBLE_DEVICES=5

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/video/ \
  --data_path gtscore_video.csv \
  --model_id bandwidth_16_16 \
  --model $model_name \
  --data video \
  --features S \
  --batch_size 256 \
  --seq_len 8 \
  --label_len 0 \
  --pred_len 8 \
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
  --train_epochs 20 \
  --inverse