
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path stock.csv \
  --model_id stock_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --seq_len $seq_len \
  --pred_len 24 \
  --enc_in 6 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'stock_$seq_len'_'24.log  
