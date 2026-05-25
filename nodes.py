import os
import torch
import folder_paths
from .pid_model import PiDModel

class PiDLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        models = folder_paths.get_filename_list("pid")
        if not models:
            models = ["No models found. Place weights in ComfyUI/models/pid/"]
        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "inference_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "scale": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("PID_MODEL",)
    RETURN_NAMES = ("pid_model",)
    FUNCTION = "load_model"
    CATEGORY = "z-image/PiD"

    def load_model(self, model_name, inference_steps, cfg_scale, scale):
        if "No models found" in model_name:
            raise RuntimeError("No PiD weights found. Download and place them in ComfyUI/models/pid/")
        model_path = folder_paths.get_full_path("pid", model_name)
        return (PiDModel(model_path, inference_steps, cfg_scale, scale),)


class PiDEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), "pid_model": ("PID_MODEL",)}}
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "z-image/PiD"

    def encode(self, images, pid_model):
        x = images.permute(0, 3, 1, 2).mul(2.0).sub(1.0)
        x = x.to(dtype=torch.bfloat16, device=pid_model.device)
        latent = pid_model.encode(x)
        return ({"samples": latent},)


class PiDDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "pid_model": ("PID_MODEL",),
                "caption": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "z-image/PiD"

    def decode(self, samples, pid_model, caption, degrade_sigma=0.0, seed=42):
        latent = samples["samples"]
        if degrade_sigma > 0:
            gen = torch.Generator(device=pid_model.device).manual_seed(seed)
            noise = torch.randn_like(latent, generator=gen)
            latent = (1.0 - degrade_sigma) * latent + degrade_sigma * noise
            
        out = pid_model.decode(latent, caption, seed=seed)
        out = out.float().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0).permute(0, 2, 3, 1)
        return (out,)


class PiDVAERoundtrip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "pid_model": ("PID_MODEL",),
                "caption": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "roundtrip"
    CATEGORY = "z-image/PiD"

    def roundtrip(self, images, pid_model, caption, degrade_sigma=0.0, seed=42):
        x = images.permute(0, 3, 1, 2).mul(2.0).sub(1.0).to(dtype=torch.bfloat16, device=pid_model.device)
        latent = pid_model.encode(x)
        if degrade_sigma > 0:
            gen = torch.Generator(device=pid_model.device).manual_seed(seed)
            noise = torch.randn_like(latent, generator=gen)
            latent = (1.0 - degrade_sigma) * latent + degrade_sigma * noise
            
        out = pid_model.decode(latent, caption, seed=seed)
        out = out.float().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0).permute(0, 2, 3, 1)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "PiDLoadModel": PiDLoadModel,
    "PiDEncode": PiDEncode,
    "PiDDecode": PiDDecode,
    "PiDVAERoundtrip": PiDVAERoundtrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDLoadModel": "Load PiD Model",
    "PiDEncode": "PiD Encode",
    "PiDDecode": "PiD Decode",
    "PiDVAERoundtrip": "PiD VAE Roundtrip",
}
