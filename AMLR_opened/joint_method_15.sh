DATA_DIR='./data/twitter2015'
#BERT_DIR='data/bert/bert-base-uncased'
IMAGE_DIR='/home/tiggers/Newdisk/lep/data/Twitrer/Twitter/twitter2015_images/'
path='output/model_checkpoint/tw15'

echo "enter sh"
image_cache_dir='/home/tiggers/Newdisk/lep/data/image_cache_dir/'

nohup ~/anaconda3/envs/lep39/bin/python -u -m run_joint_span.py \
  --num_train_epochs 40 \
  --max_seq_length 50 \
  --gpu_idx 0 \
  --do_predict False \
  --do_train True \
  --save_proportion 0.01 \
  --learning_rate 2e-5 \
  --data_dir_ner $DATA_DIR \
  --train_batch_size 32 \
  --multi_head_num 8 \
  --gradient_accumulation_steps 8 \
  --predict_batch_size 16 \
  --cache_dir $image_cache_dir \
  --image_path $IMAGE_DIR \
  --output_dir $path \
  1>$path/train.log 2>&1
echo "exit sh"