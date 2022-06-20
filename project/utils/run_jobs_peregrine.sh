#!/bin/bash
#SBATCH --job-name=dh
#SBATCH --time=5:00:00
#SBATCH --mem=64g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --array=0-5
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/deep_hashing/bin/activate

train_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_train.py"
predict_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_predict.py"

case ${SLURM_ARRAY_TASK_ID} in
0) python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 1 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
1) python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
2) python "$train_script" --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 0 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
3) python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 1 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
4) python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 2 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
5) python "$train_script" --push_kernel_size 5 --pull_kernel_size 5 --avg_kernel_size 0 --pull_inhibition_strength 3 --no-scale_the_outputs --bias --num_workers 12 --version "${SLURM_ARRAY_TASK_ID}" ;;
esac

python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt"

base_dir="/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch"
case ${SLURM_ARRAY_TASK_ID} in
0) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push3_pull3_avg0_inhibition1_scale0_bias1" ;;
1) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push3_pull3_avg0_inhibition2_scale0_bias1" ;;
2) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push3_pull3_avg0_inhibition3_scale0_bias1" ;;
3) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push5_pull5_avg0_inhibition1_scale0_bias1" ;;
4) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push5_pull5_avg0_inhibition2_scale0_bias1" ;;
5) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/push5_pull5_avg0_inhibition3_scale0_bias1" ;;
esac

#predict_script="$HOME/git_code/deep_hashing/project/DSH_push_pull_predict.py"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull3_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg0_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push3_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg3_inhibition1_scale0_bias1/checkpoints/last.ckpt"
#python "$predict_script" --model_ckpt "/data/p288722/runtime_data/deep_hashing/dsh_push_pull_scratch/push5_pull5_avg5_inhibition1_scale0_bias1/checkpoints/last.ckpt"
