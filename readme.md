```python
#### 注意必须调节 视频导入fps 25 以匹配模型特点

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
    image, _, _, video_info = VHSLoadVideo('FusionX_00006.mp4', 25, 0, 0, 0, 0, 1, None, None, 'AnimateDiff')
    _, _, _, _, _, loaded_fps, _, loaded_duration, _, _ = VHSVideoInfo(video_info)
    mmaudio_model = MMAudioModelLoader('mmaudio_large_44k_v2_fp16.safetensors', 'fp16')
    mmaudio_featureutils = MMAudioFeatureUtilsLoader('mmaudio_vae_44k_fp16.safetensors', 'mmaudio_synchformer_fp16.safetensors', 'apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors', None, '44k', 'fp16')
    audio = MMAudioSampler(mmaudio_model, mmaudio_featureutils, loaded_duration, 25, 4.5, 7, 'anime style ,This digital illustration in a vibrant, anime-inspired style depicts a vintage green tram with P.R. 13 displayed on its front, traveling down a wet, reflective railway track during a stunning sunset. The sky is ablaze with vivid orange, pink, and purple clouds, casting a warm glow over the scene. On the left, a streetlight and power poles line the track, while small buildings and a few trees are visible on the right. The trams headlights and windows reflect the colorful sky, and the tracks glisten with rain. The overall mood is nostalgic and serene, capturing a picturesque urban sunset.', '', False, True, image)
    _ = VHSVideoCombine(image, loaded_fps, 0, 'MMaudio', 'video/h264-mp4', False, False, audio, None, None)
    PreviewAudio(audio)

import os
import time
import subprocess
from pathlib import Path
import shutil
import re

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/temp'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'
SOURCE_DIR = 'Anime_Landscape_Wan_FusionX_dynamic'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 6000  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def copy_video_files():
    """Copy all .mp4 files and their corresponding .txt files from source directory to input directory"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    mp4_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    
    for file in mp4_files:
        src_path = os.path.join(SOURCE_DIR, file)
        dst_path = os.path.join(INPUT_DIR, file)
        shutil.copy2(src_path, dst_path)
        
        txt_file = os.path.splitext(file)[0] + '.txt'
        txt_src_path = os.path.join(SOURCE_DIR, txt_file)
        if os.path.exists(txt_src_path):
            txt_dst_path = os.path.join(INPUT_DIR, txt_file)
            shutil.copy2(txt_src_path, txt_dst_path)
    
    return mp4_files

def read_and_clean_text_file(video_name):
    """Read and clean text content from corresponding .txt file"""
    txt_file = os.path.splitext(video_name)[0] + '.txt'
    txt_path = os.path.join(INPUT_DIR, txt_file)
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Remove all quotes and other problematic characters
            content = re.sub(r'[\'"]', '', content)
            return content
    return ''

def generate_script(video_name, seed, prompt_text, input_dir):
    """Generate the ComfyUI script with parameters"""
    script = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

with Workflow():
    # Load video with frame rate parameter set to 25
    image, _, _, video_info = VHSLoadVideo('{video_name}', 25, 0, 0, 0, 0, 1, None, None, 'AnimateDiff')
    _, _, _, _, _, loaded_fps, _, loaded_duration, _, _ = VHSVideoInfo(video_info)
    
    # Load audio models
    mmaudio_model = MMAudioModelLoader('mmaudio_large_44k_v2_fp16.safetensors', 'fp16')
    mmaudio_featureutils = MMAudioFeatureUtilsLoader(
        'mmaudio_vae_44k_fp16.safetensors',
        'mmaudio_synchformer_fp16.safetensors',
        'apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors',
        None, '44k', 'fp16'
    )
    
    # Generate audio with cleaned prompt text
    audio = MMAudioSampler(
        mmaudio_model, 
        mmaudio_featureutils, 
        loaded_duration, 
        25,  # frames per second
        4.5,  # guidance scale
        7,    # seed
        '{prompt_text}',  # cleaned prompt text
        '',   # negative prompt
        False, 
        True, 
        image
    )
    
    # Combine video with generated audio
    _ = VHSVideoCombine(
        image, 
        loaded_fps, 
        0, 
        'MMaudio', 
        'video/h264-mp4', 
        False, 
        False, 
        audio, 
        None, 
        None
    )
    
    PreviewAudio(audio)
"""
    return script

def main():
    SEED = 661695664686456
    print(f"Copying video and text files from {SOURCE_DIR} to {INPUT_DIR}...")
    video_files = copy_video_files()
    total_videos = len(video_files)
    print(f"Copied {total_videos} videos and their corresponding text files.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for idx, video_file in enumerate(video_files):
        print(f"Processing video {idx + 1}/{total_videos}: {video_file}")
        
        # Read and clean prompt text from corresponding .txt file
        prompt_text = read_and_clean_text_file(video_file)
        print(f"Using prompt text: {prompt_text[:50]}...")  # Show first 50 chars

        video_file = str(video_file).split("/")[-1]
        # Generate script with all parameters
        script = generate_script(
            video_name=video_file,
            seed=SEED,
            prompt_text=prompt_text,
            input_dir=INPUT_DIR
        )
        
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        initial_count = get_latest_output_count()
        
        print(f"Generating audio for video with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        SEED -= 1
        print(f"Finished processing {video_file}\n")

if __name__ == "__main__":
    main()
```

# ComfyUI nodes to use [MMAudio](https://github.com/hkchengrex/MMAudio)

## WIP WIP WIP

https://github.com/user-attachments/assets/9515c0f6-cc5d-4dfe-a642-f841a1a2dba5

# Installation
Clone this repo into custom_nodes folder.

Install dependencies: pip install -r requirements.txt or if you use the portable install, run this in ComfyUI_windows_portable -folder:

python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-MMAudio\requirements.txt


Models are loaded from `ComfyUI/models/mmaudio`

Safetensors available here:

https://huggingface.co/Kijai/MMAudio_safetensors/tree/main

Nvidia bigvganv2 (used with 44k mode)

https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x

is autodownloaded to `ComfyUI/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x`
