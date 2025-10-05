import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vl_benchmark_evaluator import VLBenchmarkEvaluator
from model_config import get_model_tp, get_backend_type

def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation for VL models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')")
    parser.add_argument("--tp", type=int, required=True,
                       help="Tensor parallelism setting")
    parser.add_argument("--test_file", type=str,
                       default=None,
                       help="Path to test.jsonl file")
    parser.add_argument("--base_image_path", type=str,
                       default="data",
                       help="Base path for images")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--backend", type=str, choices=['turbomind', 'pytorch'],
                       help="Backend to use (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect backend if not specified
    if args.backend is None:
        args.backend = get_backend_type(args.model_path)
    
    print(f"Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  TP: {args.tp}")
    print(f"  Backend: {args.backend}")
    print(f"  Test file: {args.test_file}")
    print(f"  Output dir: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = VLBenchmarkEvaluator(
            model_path=args.model_path,
            base_image_path=args.base_image_path,
            tp=args.tp,
            backend=args.backend
        )
        
        # Generate output filename
        model_name = args.model_path.replace('/', '_').replace('-', '_')
        output_file = os.path.join(args.output_dir, f"benchmark_{model_name}.jsonl")
        
        # Run evaluation
        results = evaluator.evaluate_all(args.test_file, output_file)
        
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


    # CUDA_VISIBLE_DEVICES=0,1  python eval/run_benchmark.py --model_path 'OpenGVLab/InternVL3_5-14B' --tp 2 --test_file "data/test.jsonl" --output_dir "results"