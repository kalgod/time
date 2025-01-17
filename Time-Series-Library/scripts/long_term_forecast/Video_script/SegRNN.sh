export CUDA_VISIBLE_DEVICES=0

model_name=SegRNN

seq_len=5
for pred_len in 5
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/video/ \
  --data_path gtscore_video.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data video \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 5 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1
done

