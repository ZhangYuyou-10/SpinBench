# model_config.py
"""
Model configuration for A40 GPUs based on model size
"""

MODEL_TP_CONFIG = {
    # TP=1 for VLM models ≤8B
    'OpenGVLab/InternVL3-1B': 1,
    'OpenGVLab/InternVL3-2B': 1,
    'OpenGVLab/InternVL3-8B': 1,
    'OpenGVLab/InternVL3-9B': 1,
    'OpenGVLab/InternVL2_5-1B': 1,
    'OpenGVLab/InternVL2_5-2B': 1,
    'OpenGVLab/InternVL2_5-4B': 1,
    'OpenGVLab/InternVL2_5-8B': 1,  
    'OpenGVLab/InternVL2-2B': 1,
    'OpenGVLab/InternVL2-4B': 1,
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5': 1,
    'OpenGVLab/Mono-InternVL-2B': 1,
    'Qwen/Qwen2.5-VL-3B-Instruct': 1,
    'Qwen/Qwen2.5-VL-7B-Instruct': 1,
    'Qwen/Qwen2-VL-2B-Instruct': 1,
    'Qwen/Qwen2-VL-7B-Instruct': 1,
    'microsoft/Phi-3.5-vision-instruct': 1,
    'allenai/Molmo-7B-D-0924': 1,
    'openbmb/MiniCPM-V-2_6': 1,
    'internlm/internlm-xcomposer2d5-7b': 1,
    'llava-hf/llava-interleave-qwen-7b-hf': 1,
    # 'llava-hf/llava-interleave-qwen-0.5b-hf': 1,
    'llava-hf/llava-interleave-qwen-7b-dpo-hf': 1,
    'google/gemma-3-4b-it': 1,
    'OpenGVLab/InternVL3_5-1B': 1,
    'OpenGVLab/InternVL3_5-2B': 1,
    'OpenGVLab/InternVL3_5-4B': 1,
    'OpenGVLab/InternVL3_5-8B': 1,

    
    # TP=2 for VLM models ≤16B 
    'google/gemma-3-12b-it': 2,
    'OpenGVLab/InternVL3-14B': 1,
    'OpenGVLab/InternVL3_5-14B': 1,
    'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview': 2,
    
    # TP=4 for VLM models ≤32B and larger
    'google/gemma-3-27b-it': 4,
    'OpenGVLab/InternVL2_5-26B-MPO': 4,
    'OpenGVLab/InternVL2_5-26B': 4,
    'OpenGVLab/InternVL2_5-38B': 4,
    'OpenGVLab/InternVL3-38B': 4, 
    'OpenGVLab/InternVL3_5-38B': 4, 
    'OpenGVLab/InternVL2-40B': 4,
    'Qwen/Qwen2.5-VL-32B-Instruct': 4,  
    'openbmb/MiniCPM-Llama3-V-2_5': 4,  
}

def get_model_tp(model_path: str) -> int:
    """Get tensor parallelism setting for a model"""
    return MODEL_TP_CONFIG.get(model_path, 1)  # Default to tp=1

def get_gpu_count(model_path: str) -> int:
    """Get required GPU count based on model"""
    return get_model_tp(model_path)

LARGE_MEMORY_MODELS = [
    'Qwen/Qwen2.5-VL-32B-Instruct',
    'OpenGVLab/InternVL2_5-38B',
    'OpenGVLab/InternVL2-40B',
    'OpenGVLab/InternVL2-Llama3-76B-AWQ',
    'openbmb/MiniCPM-Llama3-V-2_5'
]

def is_large_memory_model(model_path: str) -> bool:
    """Check if model requires special memory handling"""
    return model_path in LARGE_MEMORY_MODELS

def is_vision_model(model_path: str) -> bool:
    """Check if model is a vision-language model"""
    vision_keywords = [
        'VL', 'Vision', 'InternVL', 'Qwen2.5-VL', 'Qwen2-VL', 'Qwen-VL', 
        'llava', 'cogvlm', 'glm-4v', 'Phi-3-vision', 'Phi-3.5-vision', 
        'Yi-VL', 'deepseek-vl', 'MiniCPM', 'Molmo', 'Llama-3.2-11B-Vision',
        'deepseek-vl2', 'MiniCPM-V'
    ]
    return any(keyword in model_path for keyword in vision_keywords)

# Available models categorized by type - Updated with comprehensive VL model list
TURBOMIND_VL_MODELS = [
    'Qwen/Qwen-VL-Chat',
    'liuhaotian/llava-v1.5-13b',
    'liuhaotian/llava-v1.6-vicuna-7b',  
    '01-ai/Yi-VL-6B',
    'deepseek-ai/deepseek-vl-1.3b-chat',
    'deepseek-ai/deepseek-vl2',
    'OpenGVLab/InternVL3-1B',
    'OpenGVLab/InternVL3-2B',
    'OpenGVLab/InternVL3-8B',
    'OpenGVLab/InternVL3-9B',
    'OpenGVLab/InternVL3-14B',
    # 'OpenGVLab/InternVL3-38B',
    'OpenGVLab/InternVL2_5-26B-MPO',
    'OpenGVLab/InternVL-Chat-V1-5',
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
    'OpenGVLab/InternVL3-2B',
    'OpenGVLab/InternVL3-8B',
    'OpenGVLab/InternVL2_5-1B',
    'OpenGVLab/InternVL2_5-8B',  
    'OpenGVLab/InternVL2_5-2B',
    'OpenGVLab/InternVL2_5-4B',
    'OpenGVLab/InternVL2_5-26B',
    'OpenGVLab/InternVL2_5-38B',
    # 'OpenGVLab/InternVL3_5-1B',
    # 'OpenGVLab/InternVL3_5-2B',
    # 'OpenGVLab/InternVL3_5-4B',
    # 'OpenGVLab/InternVL3_5-8B',
    # 'OpenGVLab/InternVL3_5-14B',
    # 'OpenGVLab/InternVL3_5-38B',
    # 'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview',
    'OpenGVLab/InternVL2-2B',
    'OpenGVLab/InternVL2-40B',
    'OpenGVLab/InternVL2-Llama3-76B-AWQ',
    'openbmb/MiniCPM-Llama3-V-2_5',
    'openbmb/MiniCPM-V-2_6',
    'allenai/Molmo-7B-D-0924'
]

PYTORCH_VL_MODELS = [
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'OpenGVLab/InternVL2_5-26B-MPO',
    'OpenGVLab/InternVL-Chat-V1-5',
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
    'OpenGVLab/InternVL3-1B',
    'OpenGVLab/InternVL3-2B',
    'OpenGVLab/InternVL3-8B',
    'OpenGVLab/InternVL3-9B',
    'OpenGVLab/InternVL3-14B',
    'OpenGVLab/InternVL3-38B',
    'OpenGVLab/InternVL2_5-1B',
    'OpenGVLab/InternVL2_5-8B',  
    'OpenGVLab/InternVL2_5-2B',
    'OpenGVLab/InternVL2_5-4B',
    'OpenGVLab/InternVL2_5-26B',
    'OpenGVLab/InternVL2_5-38B',
    'OpenGVLab/InternVL3_5-1B',
    'OpenGVLab/InternVL3_5-2B',
    'OpenGVLab/InternVL3_5-4B',
    'OpenGVLab/InternVL3_5-8B',
    'OpenGVLab/InternVL3_5-14B',
    'OpenGVLab/InternVL3_5-38B',
    'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview',
    'OpenGVLab/InternVL2-2B',
    'OpenGVLab/InternVL2-4B',
    'OpenGVLab/InternVL2-40B',
    'OpenGVLab/Mono-InternVL-2B',
    'Qwen/Qwen2-VL-2B-Instruct',
    'Qwen/Qwen2-VL-7B-Instruct',
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'Qwen/Qwen2.5-VL-32B-Instruct',  
    'THUDM/cogvlm-chat-hf',
    'THUDM/cogvlm2-llama3-chinese-chat-19B',
    'THUDM/glm-4v-9b',
    'microsoft/Phi-3-vision-128k-instruct',
    'microsoft/Phi-3.5-vision-instruct'
]

def get_backend_type(model_path: str) -> str:
    """Determine the appropriate backend for the VLM model"""
    # Special handling for problematic models
    if 'MiniCPM' in model_path:
        return 'pytorch'  # Force pytorch for MiniCPM models due to token issues
    elif model_path in TURBOMIND_VL_MODELS:
        return 'turbomind'
    elif model_path in PYTORCH_VL_MODELS:
        return 'pytorch'
    else:
        return 'pytorch'