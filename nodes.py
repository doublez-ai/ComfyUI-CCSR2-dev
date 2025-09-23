import os
import glob
import math
import time
import argparse

import numpy as np
from PIL import Image
import safetensors.torch

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)

from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from .pipelines.pipeline_ccsr import StableDiffusionControlNetPipeline
from .myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from .models.controlnet import ControlNetModel

class CCSR2_Upscale:

    sample_methods=['unipcmultistep', 'ddpm', 'dpmmultistep']
    start_point=['lr', 'noise']
    mixed_precision=["no", "fp16", "bf16"]

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pretrained_model": ("MODEL", ),
            "controlnet_model": ("MODEL", ),
            "vae": ("VAE", ),
            "image": ("IMAGE", ),
            "clip": ("CLIP", ),
            "sample_method": (s.sample_methods, {"default": "ddpm"}),
            "mixed_precision": (s.mixed_precision, {"default": "fp16"}),
            "steps": ("INT", {"default": 6, "min": 1, "max": 4096, "step": 1}),
            "t_max": ("FLOAT", {"default": 0.6667,"min": 0, "max": 1, "step": 0.01}),
            "t_min": ("FLOAT", {"default": 0.5,"min": 0, "max": 1, "step": 0.01}),
            "start_point": (s.start_point, {"default": "lr"}),
            "start_steps": ("INT", {"default": 999, "min": 1, "max": 4096, "step": 1}),
            "process_size": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 4.5,"min": 0, "max": 10, "step": 0.1}),
            "sample_times": ("INT", {"default": 1, "min": 1, "max": 99, "step": 1}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "mode": (
            [   
                'one_step',
                'multi_step',
                'multi_step_vae',
            ], {
               "default": 'multi_step_vae'
            }),
            "tile_size": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            "tile_stride": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
            "vae_tile_size_encode": ("INT", {"default": 1024, "min": 2, "max": 4096, "step": 8}),
            "vae_tile_size_decode": ("INT", {"default": 224, "min": 2, "max": 4096, "step": 8}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            
            
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "main"

    CATEGORY = "CCSR"

    def load_pipeline(self, args, accelerator, enable_xformers_memory_efficient_attention):
    
        scheduler_mapping = {
            'unipcmultistep': UniPCMultistepScheduler,
            'ddpm': DDPMScheduler,
            'dpmmultistep': DPMSolverMultistepScheduler,
        }

        try:
            scheduler_cls = scheduler_mapping[args.sample_method]
        except KeyError:
            raise ValueError(f"Invalid sample_method: {args.sample_method}")

        scheduler = scheduler_cls.from_pretrained(args.pretrained_model, subfolder="scheduler")

        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(args.pretrained_model, "feature_extractor"))
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model, subfolder="controlnet")

        vae_path = args.vae_model_path if args.vae_model_path else args.pretrained_model
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")

        # Freeze models
        for model in [vae, text_encoder, unet, controlnet]:
            model.requires_grad_(False)

        # Enable xformers if available
        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
                controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Ensure it is installed correctly.")

        # Initialize pipeline
        validation_pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if args.tile_vae:
            validation_pipeline._init_tiled_vae(
                encoder_tile_size=args.vae_encoder_tile_size,
                decoder_tile_size=args.vae_decoder_tile_size
            )

        # Set weight dtype based on mixed precision
        dtype_mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        weight_dtype = dtype_mapping.get(accelerator.mixed_precision, torch.float32)

        # Move models to accelerator device with appropriate dtype
        for model in [text_encoder, vae, unet, controlnet]:
            model.to(accelerator.device, dtype=weight_dtype)

        return validation_pipeline

    def process(self, args, enable_xformers_memory_efficient_attention=True):
        
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
        )

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the output folder creation
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("Controlnet")

        pipeline = self.load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)

        return ()

    def main(self, pretrained_model, controlnet_model, vae, image, clip, sample_method, 
                 mixed_precision, steps, t_max, t_min, start_point, start_steps, 
                 process_size, guidance_scale, sample_times, scale_by, mode, tile_size, 
                 tile_stride, vae_tile_size_encode, vae_tile_size_decode, seed):
        
        args = {"pretrained_model": pretrained_model,
                "controlnet_model": controlnet_model,
                "vae": vae,
                "image": image,
                "clip": clip,
                "sample_method": sample_method,
                "mixed_precision": mixed_precision,
                "steps": steps,
                "t_max": t_max,
                "t_min": t_min,
                "start_point": start_point,
                "start_steps": start_steps,
                "process_size": process_size,
                "guidance_scale": guidance_scale,
                "sample_times": sample_times,
                "scale_by": scale_by,
                "mode": mode,
                "tile_size": tile_size,
                "tile_stride": tile_stride,
                "vae_tile_size_encode": vae_tile_size_encode,
                "vae_tile_size_decode": vae_tile_size_decode,
                "seed": seed
                }
        
        #self.process(args)
        print("pretrained model: " + str(pretrained_model) + " of type: " + str(type(pretrained_model)))

        return (image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CCSR2_Upscale": CCSR2_Upscale
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSR2_Upscale": "CCSR2_Upscale"
}
