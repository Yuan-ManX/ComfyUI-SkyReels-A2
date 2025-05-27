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

from .src.models.transformer_a2 import A2Model 
from .src.models.pipeline_a2 import A2Pipeline 
from .src.models.utils import _crop_and_resize_pad, _crop_and_resize, write_mp4


class LoadA2Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline_path": ("STRING", {"default": "/path/to/model"}),
                "dtype": (["float32", "bfloat16"],),
                "device": ("STRING", {"default": "cuda"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("a2_model",)
    FUNCTION = "load_model"
    CATEGORY = "SkyReels-A2"

    def load_model(self, pipeline_path, dtype, device):
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map[dtype]

        # load models 
        image_encoder = CLIPVisionModel.from_pretrained(pipeline_path, subfolder="image_encoder", torch_dtype=torch.float32) 
        vae = AutoencoderKLWan.from_pretrained(pipeline_path, subfolder="vae", torch_dtype=torch.float32)

        print("load transformer...")
        model_path = os.path.join(pipeline_path, 'transformer')
        transformer = A2Model.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True)
        # transformer.save_pretrained("transformer", max_shard_size="5GB") 

        transformer.to(device, dtype=dtype) 

        a2_model = A2Pipeline.from_pretrained(pipeline_path, transformer=transformer, vae=vae, image_encoder=image_encoder, torch_dtype=dtype)

        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8)
        a2_model.scheduler = scheduler 
        a2_model.to(device)

        return (a2_model,)


class ReferenceImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refer_image_paths": ("STRING", {"multiline": True, "default": "['assets/human.png', 'assets/thing.png', 'assets/env.png']"}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("IMAGE_LIST", "IMAGE_LIST")
    RETURN_NAMES = ("clip_image_list", "vae_image_list")
    FUNCTION = "process_images"
    CATEGORY = "SkyReelsA2"

    def process_images(self, refer_image_paths, height, width, device):
        refer_images = eval(refer_image_paths)
        video_processor = VideoProcessor(vae_scale_factor=8)
        clip_image_list, vae_image_list = [], []

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

        return (clip_image_list, vae_image_list)


class A2Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {"default": "A man is holding a teddy bear in the forest."}),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "SkyReels-A2"

    def load_prompt(self, input_prompt):
        prompt = input_prompt
        return (prompt,)


class NegativePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_negative_prompt": ("STRING", {"default": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"}),
            }
        }

    RETURN_TYPES = ("NEGATIVEPROMPT",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "SkyReels-A2"

    def load_prompt(self, input_negative_prompt):
        negative_prompt = input_negative_prompt
        return (negative_prompt,)


class A2VideoGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a2_model": ("MODEL",),
                "clip_image_list": ("IMAGE_LIST",),
                "vae_image_list": ("IMAGE_LIST",),
                "prompt": ("PROMPT",),
                "negative_prompt": ("NEGATIVEPROMPT",),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "seed": ("INT", {"default": 42}),
                "guidance_scale": ("FLOAT", {"default": 5.0}),
                "num_frames": ("INT", {"default": 81}),
                "num_inference_steps": ("INT", {"default": 50}),
                "vae_combine": (["before", "after"], {"default": "before"}),
                "device": (["cuda", "cpu"], {"default": "cuda"})
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("video_tensor",)
    FUNCTION = "run_pipeline"
    CATEGORY = "SkyReels-A2"

    def run_pipeline(
        self,
        a2_model,
        clip_image_list,
        vae_image_list,
        prompt,
        negative_prompt,
        height,
        width,
        seed,
        guidance_scale,
        num_frames,
        num_inference_steps,
        vae_combine,
        device,
    ):
        generator = torch.Generator(device).manual_seed(seed)
        video_pt = a2_model(
            image_clip=clip_image_list,
            image_vae=vae_image_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
            num_inference_steps=num_inference_steps,
            vae_combine=vae_combine,
        ).frames
        return (video_pt,)
    

class CombineImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_tensor": ("TENSOR",),
                "refer_image_paths": ("STRING", {"multiline": True, "default": "['assets/human.png', 'assets/thing.png', 'assets/env.png']"}),
                "width": ("INT", {"default": 832}),
                "height": ("INT", {"default": 480}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_images",)
    FUNCTION = "assemble"
    CATEGORY = "SkyReels-A2"

    def assemble(self, video_tensor, refer_image_paths, width, height):
        batch_size = video_tensor.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = video_tensor[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
            pt_image = pt_image[12:]
            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        video_generate = batch_video_frames[0] 
        final_images = []
        for q in range(len(video_generate)): 
            frame1 = _crop_and_resize_pad(load_image(image=refer_image_paths[0]), height, width) 
            frame2 = _crop_and_resize_pad(load_image(image=refer_image_paths[1]), height, width) 
            frame3 = _crop_and_resize_pad(load_image(image=refer_image_paths[2]), height, width) 
            frame4 = Image.fromarray(np.array(video_generate[q])).convert("RGB")
            result = Image.new('RGB', (width * 4, height),color="white")
            result.paste(frame1, (0, 0)) 
            result.paste(frame2, (width, 0)) 
            result.paste(frame3, (width*2, 0)) 
            result.paste(frame4, (width*3, 0)) 
            final_images.append(np.array(result))

        return (final_images,)


class SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "output.mp4"}),
                "final_images": ("IMAGE",),
                "fps": ("INT", {"default": 15}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "SkyReels-A2"

    def save(self, video_path, final_images, fps):

        write_mp4(video_path, final_images, fps=fps)
        return ()
    
