import os
import json
import base64
import re
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm

# API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic not available. Install with: pip install anthropic")


class MultiModelVLMEvaluator:
    def __init__(self, 
                 api_key: str = None, 
                 base_image_path: str = None, 
                 model_name: str = "gpt-4o-mini",
                 anthropic_api_key: str = None):
        """
        Initialize evaluator with support for both OpenAI and Anthropic models.
        
        Args:
            api_key: OpenAI API key
            anthropic_api_key: Anthropic API key  
            base_image_path: Base path for images
            model_name: Model to use (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514")
        """
        self.base_image_path = base_image_path or ""
        self.model_name = model_name
        
        # Determine which API to use based on model name
        if "claude" in model_name.lower():
            self.api_type = "anthropic"
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package required for Claude models. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.api_type = "openai"
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package required for GPT models. Install with: pip install openai")
            self.client = OpenAI(api_key=api_key)
        
        print(f"✅ Initialized {self.model_name} evaluator ({self.api_type})")
        print(f"📁 Base image path: {self.base_image_path}")
    
    def load_test_data(self, test_file_path: str) -> List[Dict[str, Any]]:
        """Load test data from jsonl file."""
        test_data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        print(f"📝 Loaded {len(test_data)} test cases from {test_file_path}")
        return test_data
    
    def get_available_task_types(self, test_data: List[Dict[str, Any]]) -> Set[str]:
        """Get all available task types from the test data."""
        task_types = set()
        for entry in test_data:
            task_type = entry.get('metadata', {}).get('task_type', 'unknown')
            task_types.add(task_type)
        return task_types
    
    def filter_by_task_type(self, test_data: List[Dict[str, Any]], 
                           include_types: Optional[List[str]] = None,
                           exclude_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter test data by task types."""
        if include_types is None and exclude_types is None:
            return test_data
        
        filtered_data = []
        include_set = set(include_types) if include_types else None
        exclude_set = set(exclude_types) if exclude_types else set()
        
        for entry in test_data:
            task_type = entry.get('metadata', {}).get('task_type', 'unknown')
            
            if include_set is not None and task_type not in include_set:
                continue
                
            if task_type in exclude_set:
                continue
                
            filtered_data.append(entry)
        
        return filtered_data
    
    def print_task_type_summary(self, test_data: List[Dict[str, Any]], title: str = "Task Type Summary"):
        """Print a summary of task types in the data."""
        task_type_counts = {}
        for entry in test_data:
            task_type = entry.get('metadata', {}).get('task_type', 'unknown')
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        print(f"\n📊 {title}:")
        for task_type, count in sorted(task_type_counts.items()):
            print(f"  • {task_type}: {count} cases")
        print(f"  Total: {len(test_data)} cases")
    
    def encode_image(self, image_path: str) -> tuple:
        """Encode image to base64 string and detect media type."""
        full_path = os.path.join(self.base_image_path, image_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        # Detect media type from file extension
        ext = os.path.splitext(full_path)[1].lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        with open(full_path, "rb") as image_file:
            b64_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        return b64_data, media_type
    
    def prepare_openai_message(self, problem: str, image_paths: List[str]) -> List[Dict]:
        """Prepare message format for OpenAI GPT API."""
        base64_images = []
        for img_path in image_paths:
            try:
                b64_img, _ = self.encode_image(img_path)
                base64_images.append(b64_img)
            except Exception as e:
                print(f"⚠️  Warning: Failed to process image {img_path}: {e}")
                continue
        
        # Remove <image> tokens from text and clean it up
        clean_text = problem.replace('<image>', '').strip()
        clean_text = ' '.join(clean_text.split())
        clean_text = clean_text + "\nDirectly answer your choice with format: <answer>Captial letter</answer>\n "

        # Create content array: text first, then all images
        content = [{"type": "text", "text": clean_text}]
        
        for b64_img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        
        messages = [{"role": "user", "content": content}]
        print(f"📸 Prepared OpenAI message: text + {len(base64_images)} images")
        return messages
    
    def prepare_o4_message(self, problem: str, image_paths: List[str]) -> List[Dict]:
        """Prepare message format for OpenAI GPT API."""
        base64_images = []
        for img_path in image_paths:
            try:
                b64_img, _ = self.encode_image(img_path)
                base64_images.append(b64_img)
            except Exception as e:
                print(f"⚠️  Warning: Failed to process image {img_path}: {e}")
                continue
        
        # Remove <image> tokens from text and clean it up
        clean_text = problem.replace('<image>', '').strip()
        clean_text = ' '.join(clean_text.split())
        clean_text = clean_text + "\nDirectly answer your choice with format: <answer>Captial letter</answer>\n "

        # Create content array: text first, then all images
        content = [{"type": "input_text", "text": clean_text}]
        
        for b64_img in base64_images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64_img}"
            })
        
        messages = [{"role": "user", "content": content}]
        print(f"📸 Prepared OpenAI message: text + {len(base64_images)} images")
        return messages
    
    def prepare_claude_message(self, problem: str, image_paths: List[str]) -> List[Dict]:
        """Prepare message format for Claude API."""
        image_blocks = []
        for img_path in image_paths:
            try:
                b64_data, media_type = self.encode_image(img_path)
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data
                    }
                })
            except Exception as e:
                print(f"⚠️  Warning: Failed to process image {img_path}: {e}")
                continue
        
        # Remove <image> tokens from text and clean it up
        clean_text = problem.replace('<image>', '').strip()
        clean_text = ' '.join(clean_text.split())
        clean_text = clean_text + "\nDo NOT output your reasoning.\nDirectly answer your choice with format: <answer>Captial letter</answer>\n "
        
        # For Claude, images should come before text for best performance
        content = image_blocks + [{"type": "text", "text": clean_text}]
        
        messages = [{"role": "user", "content": content}]
        print(f"📸 Prepared Claude message: {len(image_blocks)} images + text")
        return messages
    


    def call_openai_api(self, messages: List[Dict]) -> tuple:
        """Make API call to OpenAI models."""
        try:
            if self.model_name != "o4-mini":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.0,
                    top_p=1.0
                )
                response_text = response.choices[0].message.content.strip()
                # Pricing table (per 1K tokens)
                pricing = {
                    "gpt-4.1": {"input": 0.002, "output": 0.008},
                    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
                    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
                    "gpt-4.5-preview": {"input": 0.075, "output": 0.15},
                    "gpt-4o-realtime-preview": {"input": 0.005, "output": 0.02},
                    "gpt-4o-audio-preview": {"input": 0.0025, "output": 0.01},
                    "gpt-4o": {"input": 0.0025, "output": 0.01},
                    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                    "gpt-4o-mini-audio-preview": {"input": 0.00015, "output": 0.0006},
                    "gpt-4o-mini-realtime-preview": {"input": 0.0006, "output": 0.0024},
                    "o1-pro": {"input": 0.15, "output": 0.6},
                    "o1": {"input": 0.015, "output": 0.06},
                    "o3-pro": {"input": 0.02, "output": 0.08},
                    "o3": {"input": 0.002, "output": 0.008},
                    "o3-deep-research": {"input": 0.01, "output": 0.04},
                    "o3-mini": {"input": 0.0011, "output": 0.0044},
                    "o1-mini": {"input": 0.0011, "output": 0.0044},
                    "o4-mini": {"input": 0.0011, "output": 0.0044},
                    "o4-mini-deep-research": {"input": 0.002, "output": 0.008},
                }

                # Get usage
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Match model
                model_key = self.model_name.lower()
                matched = False
                for name, price in pricing.items():
                    if name in model_key:
                        prompt_cost = prompt_tokens * price["input"] / 1000
                        completion_cost = completion_tokens * price["output"] / 1000
                        total_cost = prompt_cost + completion_cost
                        matched = True
                        break

                if not matched:
                    total_cost = 0
                    print(f"⚠️ No pricing found for model: {self.model_name}")

                print(f"🪙 Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, Cost: ${total_cost:.6f}")
            else:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=messages,
                    reasoning={
                            "effort": "medium",
                            # "summary": "auto"
                    },
                    max_output_tokens=5000,
                )
                if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
                    print("🪙 Ran out of tokens")
                    if response.output_text:
                        print("Partial output:", response.output_text)
                        response_text = response.output_text
                    else: 
                        print("🪙 Ran out of tokens during reasoning")
                if response.output_text:
                    print("Output:", response.output_text)
                    response_text = response.output_text
                # Get usage
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                total_tokens = response.usage.total_tokens
                prompt_cost = prompt_tokens * 0.0011 / 1000
                completion_cost = completion_tokens * 0.0044 / 1000
                total_cost = prompt_cost + completion_cost
                print(f"🪙 Tokens: {prompt_tokens}+{completion_tokens}={total_tokens}, Reasoning Tokens: {reasoning_tokens}  Cost: ${total_cost:.6f}")

            return response_text, total_cost

        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            raise e

    

    def call_claude_api(self, messages: List[Dict]) -> tuple:
        """Make API call to Claude models."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                temperature=0.0,
                messages=messages
            )

            response_text = response.content[0].text.strip()

            # Claude pricing table (per 1K tokens)
            pricing = {
                "claude-opus-4-1-20250805": {"input": 0.015, "output": 0.075},
                "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
                "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
                "claude-3-7-sonnet-20250219": {"input": 0.003, "output": 0.015},
                "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
                "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            }

            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            total_tokens = input_tokens + output_tokens

            model_key = self.model_name.lower()
            matched = False
            for name, price in pricing.items():
                if name in model_key:
                    input_cost = input_tokens * price["input"] / 1000
                    output_cost = output_tokens * price["output"] / 1000
                    total_cost = input_cost + output_cost
                    matched = True
                    break

            if not matched:
                total_cost = 0  # default if model not found
                print("⚠️ Model not found in pricing table.")

            print(f"🪙 Tokens: {input_tokens}+{output_tokens}={total_tokens}, Cost: ${total_cost:.6f}")
            return response_text, total_cost

        except Exception as e:
            print(f"❌ Claude API call failed: {e}")
            raise e

    
    def call_api(self, messages: List[Dict]) -> tuple:
        """Call the appropriate API based on model type."""
        if self.api_type == "anthropic":
            return self.call_claude_api(messages)
        else:
            return self.call_openai_api(messages)
    
    def prepare_message(self, problem: str, image_paths: List[str]) -> List[Dict]:
        """Prepare message in the appropriate format for the model."""
        if self.api_type == "anthropic":
            return self.prepare_claude_message(problem, image_paths)
        else:
            if self.model_name == "o4-mini":
                return self.prepare_o4_message(problem, image_paths)
            else:
                return self.prepare_openai_message(problem, image_paths)
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer (A, B, or C D) from the model response."""
        patterns = [
            r'<answer>\s*([ABCD])\s*</answer>',  # Match <answer>B</answer>
            r'<answer>\s*([ABCD])\s*<answer>',   # Handle malformed tags
            r'\b([ABCd])\b'  # Standalone capital letter
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[0].strip().upper()
        
        return 'UNKNOWN'
    
    def evaluate_single(self, test_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case."""
        try:
            # Prepare the message
            messages = self.prepare_message(
                test_entry['problem'], 
                test_entry['images']
            )
            
            # Call API
            response_text, cost = self.call_api(messages)
            
            # Extract answer
            model_answer = self.extract_answer(response_text)
            
            # Check if correct
            ground_truth = test_entry['answer']
            is_correct = model_answer == ground_truth
            
            # Prepare result
            result = {
                'original_entry': test_entry,
                'model_name': self.model_name,
                'api_type': self.api_type,
                'model_response': response_text,
                'extracted_answer': model_answer,
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'cost': cost,
                'task_type': test_entry.get('metadata', {}).get('task_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
            return {
                'original_entry': test_entry,
                'model_name': self.model_name,
                'api_type': self.api_type,
                'model_response': f"ERROR: {str(e)}",
                'extracted_answer': 'ERROR',
                'ground_truth': test_entry['answer'],
                'is_correct': False,
                'cost': 0,
                'task_type': test_entry.get('metadata', {}).get('task_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_all(self, test_file_path: str, output_file_path: str = None, resume: bool = True,
                    include_task_types: Optional[List[str]] = None,
                    exclude_task_types: Optional[List[str]] = None,
                    sleep_duration: float = 5.0) -> Dict[str, Any]:
        """Evaluate all test cases and save results incrementally."""
        
        # Load test data
        all_test_data = self.load_test_data(test_file_path)
        
        # Show available task types
        available_types = self.get_available_task_types(all_test_data)
        print(f"\n🏷️  Available task types: {sorted(available_types)}")
        
        # Filter by task types if specified
        test_data = self.filter_by_task_type(all_test_data, include_task_types, exclude_task_types)
        
        if len(test_data) == 0:
            print("❌ No test cases match the specified task type filters!")
            return {'summary': {}, 'results': []}
        
        # Show what we're evaluating
        if include_task_types or exclude_task_types:
            self.print_task_type_summary(all_test_data, "Original Data")
            self.print_task_type_summary(test_data, "Filtered Data (will evaluate)")
            
            filter_info = []
            if include_task_types:
                filter_info.append(f"including: {include_task_types}")
            if exclude_task_types:
                filter_info.append(f"excluding: {exclude_task_types}")
            print(f"🔍 Filtering applied: {', '.join(filter_info)}")
        
        # Prepare output file
        if output_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filter_suffix = ""
            if include_task_types:
                filter_suffix += f"_inc_{'_'.join(include_task_types)}"
            if exclude_task_types:
                filter_suffix += f"_exc_{'_'.join(exclude_task_types)}"
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
            output_file_path = f"{safe_model_name}_results_{timestamp}{filter_suffix}.jsonl"

        print("output_file_path", output_file_path, self.model_name)
        
        # Check for existing results if resume is enabled
        processed_indices = set()
        results = []
        if resume and os.path.exists(output_file_path):
            print(f"📂 Found existing results file: {output_file_path}")
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Skip summary line and load existing results
                for i, line in enumerate(lines[1:], 0):
                    if line.strip():
                        try:
                            result = json.loads(line)
                            if 'original_entry' in result:
                                # Find matching test case index
                                for j, test_entry in enumerate(test_data):
                                    if (result['original_entry'].get('images') == test_entry.get('images') and
                                        result['original_entry'].get('problem') == test_entry.get('problem')):
                                        processed_indices.add(j)
                                        results.append(result)
                                        break
                        except json.JSONDecodeError:
                            continue
                
                print(f"📋 Resuming: {len(processed_indices)} cases already completed")
            except Exception as e:
                print(f"⚠️  Could not load existing results: {e}")
                print("🔄 Starting fresh evaluation...")
                processed_indices = set()
                results = []
        
        # Initialize file with placeholder summary
        if not resume or not os.path.exists(output_file_path):
            with open(output_file_path, 'w', encoding='utf-8') as f:
                placeholder_summary = {
                    'model_name': self.model_name,
                    'api_type': self.api_type,
                    'status': 'IN_PROGRESS',
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'filter_config': {
                        'include_task_types': include_task_types,
                        'exclude_task_types': exclude_task_types,
                        'total_original_cases': len(all_test_data),
                        'total_filtered_cases': len(test_data)
                    }
                }
                f.write(json.dumps(placeholder_summary) + '\n')
        
        # Process remaining test cases
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_cost = sum(r.get('cost', 0) for r in results)
        task_type_stats = {}
        
        # Rebuild stats from existing results
        for result in results:
            task_type = result.get('task_type', 'unknown')
            if task_type not in task_type_stats:
                task_type_stats[task_type] = {'total': 0, 'correct': 0}
            task_type_stats[task_type]['total'] += 1
            if result.get('is_correct', False):
                task_type_stats[task_type]['correct'] += 1
        
        remaining_cases = [i for i in range(len(test_data)) if i not in processed_indices]
        
        print(f"🚀 Starting evaluation with {self.model_name}...")
        print(f"📊 Total cases: {len(test_data)}, Remaining: {len(remaining_cases)}")
        
        # Open file in append mode for incremental writing
        with open(output_file_path, 'a', encoding='utf-8') as f:
            try:
                for idx, i in enumerate(tqdm(remaining_cases, desc="Evaluating", initial=len(processed_indices), total=len(test_data))):
                    test_entry = test_data[i]
                    
                    # Evaluate single case
                    result = self.evaluate_single(test_entry)
                    results.append(result)
                    
                    # Write result immediately to file
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    
                    # Update statistics
                    if result['is_correct']:
                        correct_count += 1
                    total_cost += result.get('cost', 0)
                    
                    task_type = result['task_type']
                    if task_type not in task_type_stats:
                        task_type_stats[task_type] = {'total': 0, 'correct': 0}
                    task_type_stats[task_type]['total'] += 1
                    if result['is_correct']:
                        task_type_stats[task_type]['correct'] += 1
                    
                    # Print progress every 10 cases
                    if (len(results)) % 10 == 0:
                        current_accuracy = correct_count / len(results) if results else 0
                        print(f"📊 Progress: {len(results)}/{len(test_data)} cases, Accuracy: {current_accuracy:.3f}, Cost: ${total_cost:.4f}")
                    
                    # Sleep between requests to avoid rate limits
                    if idx < len(remaining_cases) - 1 and sleep_duration > 0:
                        time.sleep(sleep_duration)
            
            except KeyboardInterrupt:
                print(f"\n⚠️  Evaluation interrupted by user")
                print(f"💾 Progress saved: {len(results)}/{len(test_data)} cases completed")
                print(f"📁 Partial results saved in: {output_file_path}")
                print(f"🔄 Run again with same output file to resume from where you left off")
            
            except Exception as e:
                print(f"\n❌ Evaluation failed: {e}")
                print(f"💾 Progress saved: {len(results)}/{len(test_data)} cases completed")
                print(f"📁 Partial results saved in: {output_file_path}")
        
        # Calculate final accuracy by task type
        for task_type in task_type_stats:
            stats = task_type_stats[task_type]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Prepare final summary
        summary = {
            'model_name': self.model_name,
            'api_type': self.api_type,
            'total_cases': len(results),
            'correct_cases': correct_count,
            'overall_accuracy': correct_count / len(results) if len(results) > 0 else 0,
            'total_cost': total_cost,
            'task_type_breakdown': task_type_stats,
            'evaluation_timestamp': datetime.now().isoformat(),
            'status': 'COMPLETED' if len(results) == len(test_data) else 'PARTIAL',
            'filter_config': {
                'include_task_types': include_task_types,
                'exclude_task_types': exclude_task_types,
                'total_original_cases': len(all_test_data),
                'total_filtered_cases': len(test_data)
            }
        }
        
        # Update the summary at the beginning of the file
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(summary) + '\n')
                f.writelines(lines[1:])
        
        except Exception as e:
            print(f"⚠️  Could not update summary: {e}")
            summary_file = output_file_path.replace('.jsonl', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"📋 Summary saved separately to: {summary_file}")
        
        # Print final summary
        print(f"\n{'='*60}")
        if len(results) == len(test_data):
            print(f"🎉 {self.model_name} EVALUATION COMPLETE!")
        else:
            print(f"⏸️  {self.model_name} EVALUATION PARTIAL ({len(results)}/{len(test_data)})")
        print(f"{'='*60}")
        print(f"📊 Overall Accuracy: {summary['overall_accuracy']:.3f} ({correct_count}/{len(results)})")
        print(f"💰 Total Cost: ${total_cost:.4f}")
        print(f"📝 Cases Processed: {len(results)}/{len(test_data)}")
        
        if include_task_types or exclude_task_types:
            print(f"🔍 Filtered from {len(all_test_data)} original cases")
        
        if task_type_stats:
            print(f"\n📋 Accuracy by Task Type:")
            for task_type, stats in task_type_stats.items():
                print(f"  • {task_type}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print(f"\n💾 Results saved to: {output_file_path}")
        
        if len(results) < len(test_data):
            print(f"🔄 To resume: python {__file__} --test_file {test_file_path} --output_file {output_file_path}")
        
        return {
            'summary': summary,
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-Model VLM Evaluator (GPT + Claude)")
    parser.add_argument("--test_file", type=str, required=True,
                       help="Path to test.jsonl file")
    parser.add_argument("--base_image_path", type=str, required=True,
                       help="Base path for images")
    parser.add_argument("--output_file", type=str,
                       help="Output file path (optional)")
    parser.add_argument("--openai_api_key", type=str,
                       help="OpenAI API key (optional, uses env var if not provided)")
    parser.add_argument("--anthropic_api_key", type=str,
                       help="Anthropic API key (optional, uses env var if not provided)")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                       help="Model name (e.g., 'gpt-4o-mini', 'claude-sonnet-4-20250514')")
    parser.add_argument("--sample", type=int,
                       help="Only test first N samples (for quick testing)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing results, start fresh")
    parser.add_argument("--sleep", type=float, default=5.0,
                       help="Sleep duration between API calls (seconds)")
    
    # Task type filtering arguments
    parser.add_argument("--task_types", type=str, nargs='+',
                       help="Only evaluate specific task types")
    parser.add_argument("--exclude_task_types", type=str, nargs='+',
                       help="Exclude specific task types")
    parser.add_argument("--list_task_types", action="store_true",
                       help="List all available task types in the test file and exit")
    
    args = parser.parse_args()

    # Fallback to env vars if API keys are not provided as args
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Check if missing
    if not openai_key and "gpt" in args.model_name:
        raise ValueError("Missing OpenAI API key. Provide --openai_api_key or set OPENAI_API_KEY env var.")

    if not anthropic_key and "claude" in args.model_name:
        raise ValueError("Missing Anthropic API key. Provide --anthropic_api_key or set ANTHROPIC_API_KEY env var.")
    
    # Initialize evaluator
    evaluator = MultiModelVLMEvaluator(
        api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        base_image_path=args.base_image_path,
        model_name=args.model_name 
    )
    
    # Handle listing task types
    if args.list_task_types:
        test_data = evaluator.load_test_data(args.test_file)
        available_types = evaluator.get_available_task_types(test_data)
        evaluator.print_task_type_summary(test_data, "Available Task Types")
        print(f"\n🏷️  Task types: {sorted(available_types)}")
        return
    
    # Handle sampling for quick tests
    test_file = args.test_file
    if args.sample:
        print(f"🧪 Sample mode: Testing only first {args.sample} cases")
        with open(args.test_file, 'r') as f:
            lines = f.readlines()
        
        sample_file = f"temp_sample_{args.sample}.jsonl"
        with open(sample_file, 'w') as f:
            f.writelines(lines[:args.sample])
        
        test_file = sample_file
    
    try:
        # Run evaluation
        resume = not args.no_resume
        results = evaluator.evaluate_all(
            test_file, 
            args.output_file, 
            resume=resume,
            include_task_types=args.task_types,
            exclude_task_types=args.exclude_task_types,
            sleep_duration=args.sleep
        )
        
        print(f"\n🎯 Quick Summary:")
        print(f"   Model: {results['summary']['model_name']} ({results['summary']['api_type']})")
        print(f"   Accuracy: {results['summary']['overall_accuracy']:.1%}")
        print(f"   Total Cost: ${results['summary'].get('total_cost', 0):.4f}")
        print(f"   Status: {results['summary'].get('status', 'COMPLETED')}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
    finally:
        # Cleanup sample file
        if args.sample and os.path.exists(sample_file):
            os.remove(sample_file)


if __name__ == "__main__":
    main()


    # example use:
    # python eval/proprietary_evaluator.py --test_file "data/test.jsonl" --base_image_path "data" --model_name claude-3-5-haiku-20241022 
    # python eval/proprietary_evaluator.py --test_file "data/test.jsonl" --base_image_path "data" --model_name claude-sonnet-4-20250514
    # python eval/proprietary_evaluator.py --test_file "data/test.jsonl" --base_image_path "data" --model_name gpt-4o