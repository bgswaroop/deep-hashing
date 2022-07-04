#!/bin/bash
#SBATCH --job-name=dh
#SBATCH --time=8:00:00
#SBATCH --mem=64g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --array=0-4
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/deep_hashing/bin/activate

#train_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_train.py"
predict_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_predict.py"
logs_dir="/data/p288722/runtime_data/deep_hashing"
experiment_name="dsh_push_pull_scratch_48bit"
#common_train_args="--hash_length 48 --no-scale_the_outputs --bias --num_workers 12 --logs_dir ${logs_dir} --experiment_name ${experiment_name}"
#
#case ${SLURM_ARRAY_TASK_ID} in
#0)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#1)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#2)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#3)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#4)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#5)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#6)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#7)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#8)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#9)  python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#10) python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#11) python ${train_script} --push_kernel_size 3 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#12) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#13) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#14) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#15) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#16) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#17) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 3 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#18) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 5 --pull_inhibition_strength 0.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#19) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 5 --pull_inhibition_strength 1.0 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#20) python ${train_script} --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 5 --pull_inhibition_strength 1.5 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
#esac

corruption_types="motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
base_dir="$logs_dir/$experiment_name"
baseline_classifier_results_dir="/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch_48bit/baseline_48-bit-scratch_60epochs/results"
common_predict_args="--num_workers 12 --baseline_classifier_results_dir ${baseline_classifier_results_dir} --corruption_types $corruption_types"
#python ${predict_script} --model_ckpt "${base_dir}/baseline_48-bit-scratch_60epochs/checkpoints/last.ckpt" ${common_predict_args} --no-use_push_pull
#python ${predict_script} --model_ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_predict_args}

case ${SLURM_ARRAY_TASK_ID} in
0)
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg0_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg0_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg0_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg3_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg5_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg3_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull3_avg3_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg0_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg0_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  ;;
1)
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg0_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg3_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg3_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push3_pull5_avg3_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg0_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg0_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg0_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg3_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg3_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  ;;
2)
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg3_inhibition3/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg5_inhibition1/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/push5_pull5_avg5_inhibition2/checkpoints/last.ckpt" ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg0_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg0_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg0_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg3_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg3_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  ;;
3)
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull3_avg3_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg0_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg0_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg0_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg3_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg3_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push3_pull5_avg3_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg0_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  ;;
4)
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg0_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg0_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg3_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg3_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg3_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg5_inhibition0.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg5_inhibition1.0/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  python ${predict_script} --model_ckpt "${base_dir}/2layers_push5_pull5_avg5_inhibition1.5/checkpoints/last.ckpt" --num_push_pull_layers 2 ${common_predict_args}
  ;;
esac

#case ${SLURM_ARRAY_TASK_ID} in
#0)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg0_inhibition0.5" ;;
#1)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg0_inhibition1.0" ;;
#2)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg0_inhibition1.5" ;;
#3)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg3_inhibition0.5" ;;
#4)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg3_inhibition1.0" ;;
#5)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull3_avg3_inhibition1.5" ;;
#6)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg0_inhibition0.5" ;;
#7)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg0_inhibition1.0" ;;
#8)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg0_inhibition1.5" ;;
#9)  mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg3_inhibition0.5" ;;
#10) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg3_inhibition1.0" ;;
#11) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push3_pull5_avg3_inhibition1.5" ;;
#12) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg0_inhibition0.5" ;;
#13) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg0_inhibition1.0" ;;
#14) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg0_inhibition1.5" ;;
#15) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg3_inhibition0.5" ;;
#16) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg3_inhibition1.0" ;;
#17) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg3_inhibition1.5" ;;
#18) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg5_inhibition0.5" ;;
#19) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg5_inhibition1.0" ;;
#20) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/2layers_push5_pull5_avg5_inhibition1.5" ;;
#esac
