import os
import torch 
from PIL import Image 
import numpy as np 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from diffusers import AutoencoderKLWan
from transformers import CLIPVisionModel 
from diffusers.video_processor import VideoProcessor
from diffusers import UniPCMultistepScheduler 
from diffusers.utils import export_to_video, load_image 
from diffusers.image_processor import VaeImageProcessor

from models.transformer_a2 import A2Model 
from models.pipeline_a2 import A2Pipeline 
from models.utils import _crop_and_resize_pad, write_mp4
import hashlib
import time


TMP_FILE_PATH = os.path.join(os.path.dirname(__file__), "tmp")
if not os.path.exists(TMP_FILE_PATH):
    os.mkdir(TMP_FILE_PATH)

DEFAULT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def _crop_and_resize_pad(image, height=480, width=720):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        pad = int((((height / width) * image_width) - image_height) / 2.)
        padded_image = np.ones((image_height + pad * 2, image_width, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height + pad * 2, image_width, 3), dtype=np.uint8)
        padded_image[pad:pad+image_height, :] = image
        image = Image.fromarray(padded_image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.ones((image_height, image_width + pad * 2, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8) 
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    return image 


def _crop_and_resize(image, height=512, width=512):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width/width*height)
        top = (image_height - croped_height) // 2
        image = image[top:top+croped_height, :]
        image = Image.fromarray(image).resize((width, height))

    return image
    

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])


WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 81
GUIDANCE_SCALE = 5.0
VAE_SCALE_FACTOR_SPATIAL = 8
STEP = 50


class ModelInference:
    def __init__(self):
        self._pipeline_path =  "/maindata/data/shared/public/multimodal/ckpt/wan14B-compose"
        self._model_path = os.path.join(self._pipeline_path, "transformer")
        self._dtype = torch.bfloat16
        self._device = "cuda"
        self._image_encoder = CLIPVisionModel.from_pretrained(self._pipeline_path, subfolder="image_encoder", torch_dtype=torch.float32) 
        self._vae = AutoencoderKLWan.from_pretrained(self._pipeline_path, subfolder="vae", torch_dtype=torch.float32)
        self._transformer = A2Model.from_pretrained(self._model_path, torch_dtype=self._dtype)
        self._transformer.to(self._device, dtype=self._dtype) 

        self._pipe = A2Pipeline.from_pretrained(self._pipeline_path, transformer=self._transformer, vae=self._vae, image_encoder=self._image_encoder, torch_dtype=self._dtype)

        self._scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8)
        self._pipe.scheduler = self._scheduler 
        self._pipe.to(self._device)
        self._video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR_SPATIAL)

    def generate_video(self, ref_img_0, ref_img_1, ref_img_2, prompt, negative_prompt, seed):
        clip_image_list = []
        vae_image_list = []
        for image in [ref_img_0, ref_img_1, ref_img_2]:
            if image is None:
                continue

            image_clip = _crop_and_resize_pad(image, height=512, width=512)
            clip_image_list.append(image_clip)
            image_vae = _crop_and_resize_pad(image, height=HEIGHT, width=WIDTH)
            image_vae = self._video_processor.preprocess(image_vae, height=HEIGHT, width=WIDTH).to(memory_format=torch.contiguous_format)
            image_vae = image_vae.unsqueeze(2).to(self._device, dtype=torch.float32)
            vae_image_list.append(image_vae)

        generator = torch.Generator(self._device).manual_seed(seed)
        video_pt = self._pipe(
            image_clip=clip_image_list, 
            image_vae=vae_image_list,
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=HEIGHT, 
            width=WIDTH, 
            num_frames=NUM_FRAMES, 
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="pt",
            num_inference_steps=STEP,
            vae_combine="before",
            # vae_repeat=False,
        ).frames

        batch_size = video_pt.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = video_pt[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        video_generate = batch_video_frames[0]
        final_images = []
        for q in range(len(video_generate)): 
            frame = Image.fromarray(np.array(video_generate[q])).convert("RGB")
            final_images.append(np.array(frame))
        name_src = f"{time.time()}_{prompt}_{seed}"
        name = hashlib.md5(name_src.encode()).hexdigest()
        video_path = os.path.join(TMP_FILE_PATH, f"{name}.mp4")
        write_mp4(video_path, final_images, fps=15)
        return video_path


infer = ModelInference()



    


