import torch 
import os
from PIL import Image 
import numpy as np 
from diffusers import AutoencoderKLWan
from transformers import CLIPVisionModel 
from diffusers.video_processor import VideoProcessor
from diffusers import UniPCMultistepScheduler 
from diffusers.utils import export_to_video, load_image 
from diffusers.image_processor import VaeImageProcessor

from models.transformer_a2 import A2Model 
from models.pipeline_a2 import A2Pipeline 
from models.utils import _crop_and_resize_pad, _crop_and_resize, write_mp4


prompt = "A man is holding a teddy bear in the forest." 
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

refer_images = ['assets/human.png', 'assets/thing.png', 'assets/env.png'] 
width = 832
height = 480 
seed = 42 

# model parameters 
device = "cuda"
video_path = "output.mp4"
pipeline_path = "/path/to/model"
dtype = torch.bfloat16

# load models 
image_encoder = CLIPVisionModel.from_pretrained(pipeline_path, subfolder="image_encoder", torch_dtype=torch.float32) 
vae = AutoencoderKLWan.from_pretrained(pipeline_path, subfolder="vae", torch_dtype=torch.float32)

print("load transformer...")
model_path = os.path.join(pipeline_path, 'transformer')
transformer = A2Model.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True)
# transformer.save_pretrained("transformer", max_shard_size="5GB") 

transformer.to(device, dtype=dtype) 

pipe = A2Pipeline.from_pretrained(pipeline_path, transformer=transformer, vae=vae, image_encoder=image_encoder, torch_dtype=dtype)

scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8)
pipe.scheduler = scheduler 
pipe.to(device)


VAE_SCALE_FACTOR_SPATIAL = 8
video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL)

# prepare reference images
clip_image_list = []
vae_image_list = []
for image_id, image_path in enumerate(refer_images): 
    image = load_image(image=image_path).convert("RGB")
    # for clip 
    image_clip = _crop_and_resize_pad(image, height=512, width=512) 
    clip_image_list.append(image_clip)
    
    # for vae 
    if image_id == 0 or image_id == 1: 
        image_vae = _crop_and_resize_pad(image, height=height, width=width) # ref image
    else:
        image_vae = _crop_and_resize(image, height=height, width=width) # background image
    
    image_vae = video_processor.preprocess(image_vae, height=height, width=width).to(memory_format=torch.contiguous_format) # (1, 3, 480, 320)
    image_vae = image_vae.unsqueeze(2).to(device, dtype=torch.float32)
    vae_image_list.append(image_vae) #.to(device, dtype=dtype))

# forward
generator = torch.Generator(device).manual_seed(seed) 
video_pt = pipe(
    image_clip=clip_image_list, 
    image_vae=vae_image_list,
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    height=480, 
    width=width, 
    num_frames=81, 
    guidance_scale=5.0,
    generator=generator,
    output_type="pt",
    num_inference_steps=50,
    vae_combine="before",
).frames


# combine results
batch_size = video_pt.shape[0]
batch_video_frames = []
for batch_idx in range(batch_size):
    pt_image = video_pt[batch_idx]
    pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
    pt_image = pt_image[12:]
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    batch_video_frames.append(image_pil)

video_generate = batch_video_frames[0] 
final_images = []
for q in range(len(video_generate)): 
    frame1 = _crop_and_resize_pad(load_image(image=refer_images[0]), height, width) 
    frame2 = _crop_and_resize_pad(load_image(image=refer_images[1]), height, width) 
    frame3 = _crop_and_resize_pad(load_image(image=refer_images[2]), height, width) 
    frame4 = Image.fromarray(np.array(video_generate[q])).convert("RGB")
    result = Image.new('RGB', (width * 4, height),color="white")
    result.paste(frame1, (0, 0)) 
    result.paste(frame2, (width, 0)) 
    result.paste(frame3, (width*2, 0)) 
    result.paste(frame4, (width*3, 0)) 
    final_images.append(np.array(result))

write_mp4(video_path, final_images, fps=15) 
