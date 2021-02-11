DT=$(date '+%d-%m-%H:%M')

TOKENIZERS_PARALLELISM=true deepspeed --num_gpus=1 run_clm_scaling.py --model_type gpt2 --tokenizer_name gpt2 --dataset_name oscar --dataset_config_name unshuffled_deduplicated_br --do_train --do_eval --evaluation_strategy steps --eval_steps 200 --output_dir /tmp/test-clm/${DT} --deepspeed ds_config.json --n_layer 2 --n_embd 256 --n_head 8 --warmup_steps 100 --max_validation_points 50000
