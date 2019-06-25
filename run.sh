# init train
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_bert_on_race.py --data_dir=FAKE_RACE --bert_model=bert-base-uncased --output_dir=base_models --max_seq_length=400 --do_train --do_eval --do_lower_case --train_batch_size=48 --eval_batch_size=8 --learning_rate=2e-5 --num_train_epochs=20 --gradient_accumulation_steps=2 --fp16 --loss_scale=128

# load and train
# python3 run_bert_on_race.py --data_dir=RACE --bert_model=bert-base-uncased --output_dir=base_models --max_seq_length=400 --do_train --do_eval --do_lower_case --train_batch_size=48 --eval_batch_size=1 --learning_rate=2e-5 --load_from_epoch=5 --start_train_epoch=6 --num_train_epochs=10 --curr_global_step=18200 --gradient_accumulation_steps=3 --fp16 --loss_scale=128

# eval
# python3 run_bert_on_race.py --data_dir=RACE --bert_model=bert-base-uncased --output_dir=base_models --max_seq_length=400 --do_eval --do_lower_case --eval_batch_size=4 --load_from_epoch=20
