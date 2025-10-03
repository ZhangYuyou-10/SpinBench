import json
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


class LlavaOneVisionBenchmarkEvaluator:
    def __init__(self, model_path: str, base_image_path: str, device: int = 0):
        """
        Initialize the LLaVA-OneVision benchmark evaluator.
        
        Args:
            model_path: Path to the model (e.g., 'llava-hf/llava-onevision-qwen2-7b-ov-hf')
            base_image_path: Base path for images 
            device: CUDA device number
        """
        self.model_path = model_path
        self.base_image_path = base_image_path
        self.model_name = model_path.split('/')[-1]  # Extract model name from path
        self.device = device
        
        # Load model and processor
        print(f"Loading model: {model_path}")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Initialized evaluator with model: {self.model_name}")
    
    def load_test_data(self, test_file_path: str) -> List[Dict[str, Any]]:
        """
        Load test data from jsonl file.
        
        Args:
            test_file_path: Path to the test.jsonl file
            
        Returns:
            List of test entries
        """
        test_data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(test_data)} test cases from {test_file_path}")
        return test_data
    
    def load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            PIL Image
        """
        full_path = os.path.join(self.base_image_path, image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at {full_path}")
        
        # Load image
        image = Image.open(full_path).convert("RGB")
        return image
    
    def prepare_prompt_and_images(self, test_entry: Dict[str, Any]) -> tuple:
        """
        Prepare the prompt and images for model inference.
        
        Args:
            test_entry: Single test case from the jsonl file
            
        Returns:
            Tuple of (conversation, images_list)
        """
        problem = test_entry['problem']
        image_paths = test_entry['images']
        
        # Load images from paths
        images = []
        for img_path in image_paths:
            try:
                image = self.load_and_preprocess_image(img_path)
                images.append(image)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        # Count how many <image> tokens are in the problem
        image_token_count = problem.count('<image>')
        
        # Check if image token count matches actual image count
        if image_token_count != len(images):
            print(f"Warning: Image token count ({image_token_count}) doesn't match loaded images count ({len(images)})")
            print(f"  Problem: {problem[:100]}...")
            print(f"  Image paths: {image_paths}")
        
        # For LLaVA-OneVision, we need to build the conversation format
        # Replace <image> tokens with {"type": "image"} placeholders
        content = []
        
        # Split the problem text by <image> tokens
        text_parts = problem.split('<image>')
        
        # Build the content by interleaving text and image placeholders
        for i, text_part in enumerate(text_parts):
            if text_part.strip():  # Add non-empty text parts
                content.append({"type": "text", "text": text_part.strip()})
            
            # Add image placeholder if we have one and it's not the last text part
            if i < len(images):
                content.append({"type": "image"})
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": content,
            },
        ]
        
        return conversation, images
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer (A, B, C, or D) from the model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Extracted answer letter or 'UNKNOWN' if not found
        """
        # Look for answer in <answer> tags first
        answer_pattern = r'<answer>\s*([ABCDYN])\s*</answer>'
        matches = re.findall(answer_pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
        
        # Look for single capital letters A, B, C, or D
        # Try different patterns to be robust
        patterns = [
            r'\b([ABCDYN])\b',  # Single letter surrounded by word boundaries
            r'answer[:\s]+([ABCDYN])',  # "answer: A" or "answer A"
            r'([ABCDYN])(?:\.|$)',  # A followed by period or end of string
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        # If no clear pattern found, return UNKNOWN
        return 'UNKNOWN'
    
    def evaluate_single(self, test_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test case.
        
        Args:
            test_entry: Single test case
            
        Returns:
            Result dictionary with original entry, model response, and evaluation metrics
        """
        try:
            # Prepare prompt and images
            conversation, images = self.prepare_prompt_and_images(test_entry)
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(
                images=images, 
                text=prompt, 
                return_tensors='pt'
            ).to(self.device, torch.float16)
            
            # Generate response
            output = self.model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False,
                temperature=0.0
            )
            
            # Extract only the newly generated tokens (exclude input)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output[0][input_length:]
            
            # Decode only the model's response
            response_text = self.processor.decode(new_tokens, skip_special_tokens=True)
            
            # Extract model's answer
            model_answer = self.extract_answer(response_text)
            
            # Check if correct
            ground_truth = test_entry['answer']
            is_correct = model_answer == ground_truth
            
            # Prepare result
            result = {
                'original_entry': test_entry,
                'model_name': self.model_name,
                'model_response': response_text,
                'extracted_answer': model_answer,
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'task_type': test_entry.get('metadata', {}).get('task_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing test case: {e}")
            return {
                'original_entry': test_entry,
                'model_name': self.model_name,
                'model_response': f"ERROR: {str(e)}",
                'extracted_answer': 'ERROR',
                'ground_truth': test_entry['answer'],
                'is_correct': False,
                'task_type': test_entry.get('metadata', {}).get('task_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_task_name(self, task_name):
        """Parse task name to extract dataset, task type, and prompt style"""
        parts = task_name.lower().split('_')

        dataset = 'unknown'
        remaining = task_name

        if task_name.startswith('face_'):
            dataset = 'face'
            remaining = task_name[5:]
        elif task_name.startswith('car_'):
            dataset = 'car'
            remaining = task_name[4:]
        elif task_name.startswith('object_'):
            dataset = 'object'
            remaining = task_name[7:]
        elif task_name.startswith('infinigen_'):
            dataset = 'infinigen'
            remaining = task_name[10:]

        # Extract task type
        if 'identity' in remaining:
            task_type = 'identity'
        elif 'rotation_classification' in remaining:
            task_type = 'rotation_classification'
        elif 'rotation_imagination' in remaining:
            task_type = 'rotation_imagination'
        elif 'spatial_relation_grounding' in remaining:
            task_type = 'spatial_relation_grounding'
        elif 'spatial_relation_transformation' in remaining:
            task_type = 'spatial_relation_transformation'
        elif 'canonical' in remaining or 'view_selection' in remaining:
            task_type = 'view_selection'
        elif 'mental_rotation' in remaining:
            task_type = 'mental_rotation'
        elif 'canonical_view_select' in remaining:
            task_type = 'canonical_view_select'
        else:
            task_type = 'unknown'

        # Extract prompt style
        prompt_style = 'default'
        details = []
        known_styles = ['imagefirst', 'textfirst', 'interleaved', 'own_perspective', 'viewer_perspective']
        for style in known_styles:
            if style in remaining:
                prompt_style = style

        # Additional details
        for d in ['left', 'right', 'back', 'front', 'partial_occlusion', 'full_occlusion', 'no_occlusion', 'w_premise', 'wo_premise']:
            if d in remaining:
                details.append(d)

        return {
            'dataset': dataset,
            'task_type': task_type,
            'prompt_style': prompt_style,
            'details': details,
            'full_name': task_name
        }

    def _organize_results(self, task_type_stats):
        """Organize results by dataset and task type with prompt style aggregation"""
        organized = {}
        
        for task_name, stats in task_type_stats.items():
            parsed = self._parse_task_name(task_name)
            dataset = parsed['dataset']
            task_type = parsed['task_type']
            prompt_style = parsed['prompt_style']
            
            # Initialize nested structure
            if dataset not in organized:
                organized[dataset] = {}
            if task_type not in organized[dataset]:
                organized[dataset][task_type] = {
                    'by_prompt_style': {},
                    'all_samples': []
                }
            
            # Store by prompt style
            if prompt_style not in organized[dataset][task_type]['by_prompt_style']:
                organized[dataset][task_type]['by_prompt_style'][prompt_style] = []
            
            organized[dataset][task_type]['by_prompt_style'][prompt_style].append({
                'task_name': task_name,
                'stats': stats,
                'details': parsed['details']
            })
            
            # Store all samples for overall calculation
            organized[dataset][task_type]['all_samples'].append(stats)
        
        return organized

    def _calculate_aggregated_stats(self, samples_list):
        """Calculate aggregated statistics from a list of sample stats"""
        total_correct = sum(s['correct'] for s in samples_list)
        total_samples = sum(s['total'] for s in samples_list)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': total_correct,
            'total': total_samples,
            'count': len(samples_list)
        }

    def _print_enhanced_results(self, summary, task_type_stats, model_name, output_file_path, correct_count, total_results):
        """Print enhanced evaluation results with categorization and grouping"""
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f} ({correct_count}/{total_results})")
        print(f"Total Test Cases: {total_results}")
        
        # Organize results
        organized = self._organize_results(task_type_stats)
        
        # Print results by dataset and task type
        print(f"\nDETAILED RESULTS BY DATASET AND TASK TYPE:")
        print(f"{'='*80}")
        
        dataset_order = ['object', 'car', 'face', 'infinigen']  # Preferred order
        task_order = ['identity', 'rotation_classification', 'rotation_imagination', 'view_selection', 'spatial_relationship', 'mental_rotation', 'canonical_view_select']
        
        dataset_summaries = {}
        
        for dataset in dataset_order:
            if dataset not in organized:
                continue
                
            print(f"\n{dataset.upper()} DATASET:")
            print(f"{'-'*50}")
            
            dataset_all_samples = []
            
            for task_type in task_order:
                if task_type not in organized[dataset]:
                    continue
                    
                task_data = organized[dataset][task_type]
                
                # Calculate overall task accuracy
                overall_stats = self._calculate_aggregated_stats(task_data['all_samples'])
                dataset_all_samples.extend(task_data['all_samples'])
                
                print(f"\n  {task_type.replace('_', ' ').title()}:")
                print(f"    Overall: {overall_stats['accuracy']:.3f} ({overall_stats['correct']}/{overall_stats['total']}) - {overall_stats['count']} variants")
                
                # Print by prompt style if there are multiple styles
                if len(task_data['by_prompt_style']) > 1:
                    print(f"    By prompt style:")
                    for prompt_style, samples in task_data['by_prompt_style'].items():
                        style_stats = self._calculate_aggregated_stats([s['stats'] for s in samples])
                        print(f"      • {prompt_style}: {style_stats['accuracy']:.3f} ({style_stats['correct']}/{style_stats['total']}) - {style_stats['count']} variants")
                        
                        # Show individual variants if there are multiple
                        if len(samples) > 1:
                            for sample in samples:
                                details_str = ", ".join(sample['details']) if sample['details'] else ""
                                if details_str:
                                    details_str = f" ({details_str})"
                                print(f"        - {sample['task_name']}: {sample['stats']['accuracy']:.3f}{details_str}")
                else:
                    # Single prompt style, show individual tasks if multiple
                    prompt_style = list(task_data['by_prompt_style'].keys())[0]
                    samples = task_data['by_prompt_style'][prompt_style]
                    if len(samples) > 1:
                        print(f"    Breakdown:")
                        for sample in samples:
                            details_str = ", ".join(sample['details']) if sample['details'] else ""
                            if details_str:
                                details_str = f" ({details_str})"
                            print(f"      • {sample['task_name']}: {sample['stats']['accuracy']:.3f} ({sample['stats']['correct']}/{sample['stats']['total']}){details_str}")
            
            # Dataset summary
            dataset_summary = self._calculate_aggregated_stats(dataset_all_samples)
            dataset_summaries[dataset] = dataset_summary
            print(f"\n  {dataset.upper()} DATASET SUMMARY: {dataset_summary['accuracy']:.3f} ({dataset_summary['correct']}/{dataset_summary['total']})")
        
        # Overall summary by dataset
        print(f"\nDATASET COMPARISON:")
        print(f"{'-'*50}")
        for dataset in dataset_order:
            if dataset in dataset_summaries:
                stats = dataset_summaries[dataset]
                print(f"  {dataset.upper():>8}: {stats['accuracy']:.3f} ({stats['correct']:>3}/{stats['total']:>3})")
        
        print(f"\nResults saved to: {output_file_path}")
        
        return organized, dataset_summaries

    def evaluate_all(self, test_file_path: str, output_file_path: str = None) -> Dict[str, Any]:
        """
        Evaluate all test cases and save results.
        
        Args:
            test_file_path: Path to test.jsonl file
            output_file_path: Path to save results (optional)
            
        Returns:
            Summary statistics and all results
        """
        # Load test data
        test_data = self.load_test_data(test_file_path)
        
        # Process all test cases
        results = []
        correct_count = 0
        task_type_stats = {}
        
        print("Processing test cases...")
        try:
            for test_entry in tqdm(test_data):
                result = self.evaluate_single(test_entry)
                results.append(result)
                
                # Update statistics
                if result['is_correct']:
                    correct_count += 1
                
                task_type = result['task_type']
                if task_type not in task_type_stats:
                    task_type_stats[task_type] = {'total': 0, 'correct': 0}
                task_type_stats[task_type]['total'] += 1
                if result['is_correct']:
                    task_type_stats[task_type]['correct'] += 1
            
            print("All test cases processed successfully!")
            
        except Exception as e:
            print(f"Warning: Exception during processing: {e}")
            print("Continuing with partial results...")
        
        # Calculate accuracy by task type
        for task_type in task_type_stats:
            stats = task_type_stats[task_type]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Prepare summary
        summary = {
            'model_name': self.model_name,
            'total_cases': len(results),  # Use actual results count
            'correct_cases': correct_count,
            'overall_accuracy': correct_count / len(results) if len(results) > 0 else 0,
            'task_type_breakdown': task_type_stats,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        if output_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_path = f"evaluation_results_{self.model_name}_{timestamp}.jsonl"
        
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # Write summary as first line
                f.write(json.dumps(summary) + '\n')
                # Write all results
                for result in results:
                    f.write(json.dumps(result) + '\n')
            print(f"Results saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        # Print enhanced results
        organized_results, dataset_summaries = self._print_enhanced_results(
            summary, task_type_stats, self.model_name, output_file_path, correct_count, len(results)
        )
        
        return {
            'summary': summary,
            'results': results
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate LLaVA-OneVision model on vision benchmark')
    parser.add_argument('--model_path', type=str, default='llava-hf/llava-onevision-qwen2-7b-ov-hf', 
                        help='Path to the LLaVA-OneVision model')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to the test.jsonl file')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--base_image_dir', type=str, required=True,
                        help='Base directory containing images')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device number')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output file path
    model_name = args.model_path.split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"evaluation_results_{model_name}_{timestamp}.jsonl")
    
    # Initialize evaluator
    evaluator = LlavaOneVisionBenchmarkEvaluator(
        model_path=args.model_path,
        base_image_path=args.base_image_dir,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(
        test_file_path=args.test_file,
        output_file_path=output_file
    )
    
    print(f"\nEvaluation completed!")
    print(f"Overall accuracy: {results['summary']['overall_accuracy']:.3f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 python eval/run_llavaone.py --test_file "data/test.jsonl" --output_dir "./results" --base_image_dir "data"  --model_path "llava-hf/llava-onevision-qwen2-7b-ov-hf" 