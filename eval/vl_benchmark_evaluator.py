import json
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime
import base64
from openai import OpenAI
from PIL import Image
import io

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


class VLBenchmarkEvaluator:
    def __init__(self, model_path: str, base_image_path: str, tp: int = 4, backend: str = 'turbomind'):
        """
        Initialize the benchmark evaluator.
        
        Args:
            model_path: Path to the model (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')
            base_image_path: Base path for images)
            tp: Tensor parallelism for model inference
            backend: Backend to use ('turbomind' or 'pytorch')
        """
        self.model_path = model_path
        self.base_image_path = base_image_path
        self.model_name = model_path.split('/')[-1]  # Extract model name from path
        self.backend = backend


        self.gen_config = GenerationConfig(
            top_p=1.0,          # 
            temperature=0.0,    # Deterministic sampling 
            max_new_tokens=128, # 
        )

        # Initialize the pipeline with appropriate backend
        try:
            if backend == 'turbomind':
                engine_config = TurbomindEngineConfig(tp=tp)
                self.pipe = pipeline(model_path, backend_config=engine_config)
            else:  # pytorch backend
                # For PyTorch backend, let lmdeploy auto-detect the backend
                # but we can still pass tp through a PytorchEngineConfig if needed
                if tp > 1:
                    try:
                        from lmdeploy import PytorchEngineConfig

                        engine_config = PytorchEngineConfig(tp=tp)
                        self.pipe = pipeline(model_path, backend_config=engine_config)
                        print(f"✅ Initialized with PytorchEngineConfig (tp={tp})")
                    except Exception as config_error:
                        print(f"PytorchEngineConfig failed: {config_error}")
                        print(f"Trying TurbomindEngineConfig with tp={tp} as fallback...")
                        engine_config = TurbomindEngineConfig(tp=tp)
                        self.pipe = pipeline(model_path, backend_config=engine_config)
                        print(f"✅ Fallback to TurbomindEngineConfig successful (tp={tp})")
                else:
                    self.pipe = pipeline(model_path)
        except Exception as e:
            print(f"Failed to initialize with {backend} backend: {e}")
            print("Falling back to automatic backend detection...")
            # Final fallback: try turbomind first for multi-GPU models
            if tp > 1:
                try:
                    print(f"Trying TurbomindEngineConfig as final fallback (tp={tp})...")
                    engine_config = TurbomindEngineConfig(tp=tp)
                    self.pipe = pipeline(model_path, backend_config=engine_config)
                    print(f"✅ Final fallback to TurbomindEngineConfig successful (tp={tp})")
                except Exception as final_error:
                    print(f"TurbomindEngineConfig also failed: {final_error}")
                    print("❌ All configurations with tp failed!")
                    raise final_error
            else:
                self.pipe = pipeline(model_path)
        
        print(f"Initialized evaluator with model: {self.model_name} (target backend: {backend}, tp: {tp})")
    
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
    
    def prepare_prompt_and_images(self, test_entry: Dict[str, Any]) -> tuple:
        """
        Prepare the prompt and images for model inference.
        
        Args:
            test_entry: Single test case from the jsonl file
            
        Returns:
            Tuple of (prompt_with_tokens, images_list)
        """
        problem = test_entry['problem']
        image_paths = test_entry['images']
        
        # Count how many <image> tokens are in the problem
        image_token_count = problem.count('<image>')
        
        # Load images from paths
        images = []
        for img_path in image_paths:
            full_path = os.path.join(self.base_image_path, img_path)
            if os.path.exists(full_path):
                images.append(load_image(full_path))
            else:
                print(f"Warning: Image not found at {full_path}")
                continue
        
        # Check if image token count matches actual image count
        if image_token_count != len(images):
            print(f"Warning: Image token count ({image_token_count}) doesn't match loaded images count ({len(images)})")
            print(f"  Problem: {problem[:100]}...")
            print(f"  Image paths: {image_paths}")
        
        if "Qwen" in self.model_name or "InternVL" in self.model_name or "Molmo" in self.model_name:
        # if "Qwen" in self.model_name or "InternVL" in self.model_name or "Molmo" in self.model_name:
            # print("Replace <image> tokens with IMAGE_TOKEN for lmdeploy")
            prompt_with_tokens = problem.replace('<image>', IMAGE_TOKEN)
        else:
            prompt_with_tokens = problem.replace('<image>', "")
        # prompt_with_tokens = prompt_with_tokens + "\nALWAYS output in this FORMAT: <reasoning>Your analysis</reasoning>\n <answer>Captial letter</answer>\n "
        
        return prompt_with_tokens, images
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer (A, B, or C) from the model response.
        
        Args:
            response: Raw model response
            
        Returns:
            Extracted answer letter or 'UNKNOWN' if not found
        """
        # Look for single capital letters A, B, or C
        patterns = [
            r'<answer>\s*([ABCDYN])\s*</answer>',  # Match <answer>B</answer>
            r'\b([ABCDYN])\b',  # Single letter surrounded by word boundaries
            # r'answer[:\s]+([ABC])',  # "answer: A" or "answer A"
            # r'([ABC])(?:\.|$)',  # A followed by period or end of string
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        # If no clear pattern found, return the response for manual inspection
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
            prompt, images = self.prepare_prompt_and_images(test_entry)
                        
            
            response = self.pipe((prompt, images), gen_config=self.gen_config)  
            
            # Extract response text - handle different response formats
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            elif isinstance(response, list) and len(response) > 0:
                response_text = str(response[0])
            else:
                response_text = str(response)
            
            # Extract model's answer
            model_answer = self.extract_answer(response_text)
            
            # Check if correct
            ground_truth = test_entry['answer']
            is_correct = model_answer == ground_truth
            
            # Prepare result
            result = {
                'original_entry': test_entry,
                'model_name': self.model_name,
                'model_response': response.text if hasattr(response, 'text') else str(response),
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
        print(f"🎉 EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f} ({correct_count}/{total_results})")
        print(f"Total Test Cases: {total_results}")
        
        # Organize results
        organized = self._organize_results(task_type_stats)
        
        # Print results by dataset and task type
        print(f"\n📊 DETAILED RESULTS BY DATASET AND TASK TYPE:")
        print(f"{'='*80}")
        
        dataset_order = ['object', 'car', 'face', 'infinigen']  # Preferred order
        task_order = ['identity', 'rotation_classification', 'rotation_imagination', 'view_selection', 'spatial_relationship']
        
        dataset_summaries = {}
        
        for dataset in dataset_order:
            if dataset not in organized:
                continue
                
            print(f"\n🎯 {dataset.upper()} DATASET:")
            print(f"{'-'*50}")
            
            dataset_all_samples = []
            
            for task_type in task_order:
                if task_type not in organized[dataset]:
                    continue
                    
                task_data = organized[dataset][task_type]
                
                # Calculate overall task accuracy
                overall_stats = self._calculate_aggregated_stats(task_data['all_samples'])
                dataset_all_samples.extend(task_data['all_samples'])
                
                print(f"\n  📋 {task_type.replace('_', ' ').title()}:")
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
            print(f"\n  🎯 {dataset.upper()} DATASET SUMMARY: {dataset_summary['accuracy']:.3f} ({dataset_summary['correct']}/{dataset_summary['total']})")
        
        # Overall summary by dataset
        print(f"\n📈 DATASET COMPARISON:")
        print(f"{'-'*50}")
        for dataset in dataset_order:
            if dataset in dataset_summaries:
                stats = dataset_summaries[dataset]
                print(f"  {dataset.upper():>8}: {stats['accuracy']:.3f} ({stats['correct']:>3}/{stats['total']:>3})")
        
        # Task type comparison across datasets
        print(f"\n📈 TASK TYPE COMPARISON:")
        print(f"{'-'*50}")
        
        task_type_summaries = {}
        for task_type in task_order:
            all_task_samples = []
            for dataset in organized.values():
                if task_type in dataset:
                    all_task_samples.extend(dataset[task_type]['all_samples'])
            
            if all_task_samples:
                task_summary = self._calculate_aggregated_stats(all_task_samples)
                task_type_summaries[task_type] = task_summary
                print(f"  {task_type.replace('_', ' ').title():>15}: {task_summary['accuracy']:.3f} ({task_summary['correct']:>3}/{task_summary['total']:>3})")
        
        # Prompt style analysis
        print(f"\n📈 PROMPT STYLE ANALYSIS:")
        print(f"{'-'*50}")
        
        prompt_style_summaries = {}
        for dataset_data in organized.values():
            for task_data in dataset_data.values():
                for prompt_style, samples in task_data['by_prompt_style'].items():
                    if prompt_style not in prompt_style_summaries:
                        prompt_style_summaries[prompt_style] = []
                    prompt_style_summaries[prompt_style].extend([s['stats'] for s in samples])
        
        for prompt_style, samples in prompt_style_summaries.items():
            if len(samples) > 1:  # Only show if there are multiple samples
                style_summary = self._calculate_aggregated_stats(samples)
                print(f"  {prompt_style:>15}: {style_summary['accuracy']:.3f} ({style_summary['correct']:>3}/{style_summary['total']:>3}) - {style_summary['count']} tasks")
        
        print(f"\n📁 Results saved to: {output_file_path}")
        
        return organized, dataset_summaries, task_type_summaries, prompt_style_summaries


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
            
            print("✓ All test cases processed successfully!")
            
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
            print(f"✓ Results saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        organized_results, dataset_summaries, task_summaries, prompt_summaries = self._print_enhanced_results(
                    summary, task_type_stats, self.model_name, output_file_path, correct_count, len(results)
                )
        
        print(f"\n📁 Results saved to: {output_file_path}")
        
        # Try to cleanup pipeline gracefully to avoid the async error
        try:
            if hasattr(self.pipe, 'close'):
                self.pipe.close()
            elif hasattr(self.pipe, 'engine') and hasattr(self.pipe.engine, 'close'):
                self.pipe.engine.close()
        except Exception as cleanup_error:
            print(f"Note: Pipeline cleanup warning (this is usually harmless): {cleanup_error}")
        
        return {
            'summary': summary,
            'results': results
        }

