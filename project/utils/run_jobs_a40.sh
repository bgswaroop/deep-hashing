#!/bin/bash

predict_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_predict.py"
baseline_classifier_results_dir="/data/p288722/runtime_data/deep_hashing/dsh_push_pull/48-bit-finetune-without-push-pull/results/"
python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull/48-bit-finetune-without-push-pull/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir} --no-use_push_pull
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg0_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg0_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg0_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg0_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg0_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg0_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition1_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition2_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition3_scale0_bias1/checkpoints/last.ckpt" --num_workers 32 --baseline_classifier_results_dir ${baseline_classifier_results_dir}


train_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_train.py"
logs_dir="/data/p288722/runtime_data/deep_hashing"
experiment_name="dsh_push_pull_scratch_12bit"
common_train_args="--hash_length 12 --no-scale_the_outputs --bias --num_workers 32 --logs_dir ${logs_dir} --experiment_name ${experiment_name}"
python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 "${common_train_args}"

#python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 5 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 5 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 32
#python "$train_script" --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 32

#predict_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_predict.py"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/48-bit-scratch_30-60epochs/checkpoints/last.ckpt" --no-use_push_pull
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition2_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition3_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition2_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition3_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition2_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition3_scale0_bias1/checkpoints/last.ckpt"
