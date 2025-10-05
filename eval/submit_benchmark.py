# submit_benchmark.py - Convenient wrapper to submit benchmark jobs

# List all VLM models 
# python eval/submit_benchmark.py --list-models

# Test by model name(s)
# python eval/submit_benchmark.py --model "Qwen/Qwen2.5-VL-32B-Instruct"

# Test by GPU requirement
# python eval/submit_benchmark.py --tp 1  # all single-GPU VLMs
# python eval/submit_benchmark.py --tp 2  # all dual-GPU VLMs  


import os
import sys
import subprocess
import argparse
from pathlib import Path
from model_config import get_model_tp, get_gpu_count, MODEL_TP_CONFIG, is_vision_model

def submit_job(model_path, test_file, output_dir="results"):
    """Submit a benchmark job for a specific model"""
    
    # Get required GPU count
    gpu_count = get_gpu_count(model_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Prepare SLURM command
    slurm_cmd = [
        "sbatch",
        f"--gres=gpu:{gpu_count}",
        "deploy_model.slurm",
        model_path,
        str(gpu_count),
        test_file,
        output_dir
    ]
    
    print(f"Submitting job for model: {model_path}")
    print(f"GPU count: {gpu_count}")
    print(f"Command: {' '.join(slurm_cmd)}")
    
    try:
        result = subprocess.run(slurm_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully! Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"stderr: {e.stderr}")
        return None

def submit_multiple_models(models, test_file, output_dir="results"):
    """Submit benchmark jobs for multiple models"""
    
    job_ids = []
    
    for model in models:
        if model in MODEL_TP_CONFIG:
            job_id = submit_job(model, test_file, output_dir)
            if job_id:
                job_ids.append((model, job_id))
        else:
            print(f"Warning: Model {model} not found in configuration, skipping...")
    
    return job_ids

def list_available_models():
    """List all available VLM models with their configurations"""
    print("Available VLM models and their configurations:")
    print("-" * 80)
    
    models_by_tp = {}
    for model, tp in MODEL_TP_CONFIG.items():
        if tp not in models_by_tp:
            models_by_tp[tp] = []
        models_by_tp[tp].append(model)
    
    for tp in sorted(models_by_tp.keys()):
        print(f"\nTP={tp} ({tp} GPU{'s' if tp > 1 else ''}):")
        for model in sorted(models_by_tp[tp]):
            print(f"  {model}")
    
    print(f"\nTotal VLM models available: {len(MODEL_TP_CONFIG)}")

def main():
    parser = argparse.ArgumentParser(description="Submit VLM benchmark jobs for vision-language models")
    parser.add_argument("--model", type=str, help="Single VLM model to test")
    parser.add_argument("--models", nargs="+", help="Multiple VLM models to test")
    parser.add_argument("--models-file", type=str, help="File containing list of VLM models (one per line)")
    parser.add_argument("--test-file", type=str, 
                       default="data/test.jsonl",
                       help="Path to test.jsonl file")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available VLM models and their configurations")
    parser.add_argument("--tp", type=int, choices=[1, 2, 4],
                       help="Only test VLM models with specific TP setting")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Determine which models to test
    models_to_test = []
    
    if args.model:
        models_to_test = [args.model]
    elif args.models:
        models_to_test = args.models
    elif args.models_file:
        with open(args.models_file, 'r') as f:
            models_to_test = [line.strip() for line in f if line.strip()]
    else:
        # Test all available VLM models
        models_to_test = list(MODEL_TP_CONFIG.keys())
    
    # Apply TP filter if specified
    if args.tp:
        models_to_test = [m for m in models_to_test if get_model_tp(m) == args.tp]
    
    if not models_to_test:
        print("No VLM models to test!")
        return
    
    print(f"Planning to test {len(models_to_test)} VLM models:")
    for model in models_to_test:
        tp = get_model_tp(model)
        print(f"  {model} (TP={tp})")
    
    # Confirm before submitting
    response = input(f"\nSubmit {len(models_to_test)} VLM benchmark jobs? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Submit jobs
    job_ids = submit_multiple_models(models_to_test, args.test_file, args.output_dir)
    
    print(f"\nSubmitted {len(job_ids)} VLM benchmark jobs:")
    for model, job_id in job_ids:
        print(f"  {model}: {job_id}")
    
    # Create monitoring script
    monitor_script = f"""#!/bin/bash
# Monitor submitted VLM benchmark jobs
echo "Monitoring {len(job_ids)} VLM benchmark jobs:"
{chr(10).join([f'echo "Job {job_id} ({model}): $(squeue -j {job_id} -h -o %T || echo COMPLETED)"' for model, job_id in job_ids])}
"""
    
    with open("monitor_jobs.sh", "w") as f:
        f.write(monitor_script)
    os.chmod("monitor_jobs.sh", 0o755)
    
    print(f"\nUse './monitor_jobs.sh' to check job status")
    print(f"VLM benchmark results will be saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()
