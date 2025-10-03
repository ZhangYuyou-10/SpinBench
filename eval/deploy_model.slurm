#!/bin/bash
# deploy_model.slurm - Dynamic SLURM script for model testing

#SBATCH --job-name=lmdeploy
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_48


source ~/.bashrc

# Arguments will be passed: MODEL_PATH GPU_COUNT TEST_FILE OUTPUT_DIR
MODEL_PATH=$1
GPU_COUNT=$2
TEST_FILE=$3
OUTPUT_DIR=$4

if [ -z "$MODEL_PATH" ] || [ -z "$GPU_COUNT" ] || [ -z "$TEST_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch --gres=gpu:$GPU_COUNT deploy_model.slurm MODEL_PATH GPU_COUNT TEST_FILE OUTPUT_DIR"
    exit 1
fi

echo "Starting benchmark for model: $MODEL_PATH"
echo "Using $GPU_COUNT GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

conda activate spin

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | head -n $GPU_COUNT | tr '\n' ',' | sed 's/,$//')

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the benchmark
python run_benchmark.py --model_path "$MODEL_PATH" --tp $GPU_COUNT --test_file "$TEST_FILE" --output_dir "$OUTPUT_DIR"

echo "Benchmark completed for $MODEL_PATH"