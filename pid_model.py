import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

class PiDModel:
    def __init__(self, model_path: str, inference_steps: int = 4, cfg_scale: float = 1.0, scale: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_steps = inference_steps
        self.cfg_scale = cfg_scale
        self.scale = scale
        self.model_path = model_path
        
        fname = os.path.basename(model_path).lower()
        is_flux2 = any(k in fname for k in ["flux2", "klein"])
        
        self.vae_compression = 16 if is_flux2 else 8
        self.in_channels = 128 if is_flux2 else 16
        self.backbone = "flux2" if is_flux2 else "flux"
        
        self.state_dict = self._load_weights()
        self._build_projector()

    def _load_weights(self):
        if self.model_path.endswith(".safetensors"):
            sd = load_file(self.model_path, device=str(self.device))
        else:
            sd = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
        for key in ["state_dict", "model", "ema", "module"]:
            if key in sd:
                sd = sd[key]
                break
        return sd

    def _build_projector(self):
        # Создаём слой сразу на нужном устройстве и в bfloat16
        layer = torch.nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=False)
        # Безопасная инициализация: равномерное усреднение всех каналов
        with torch.no_grad():
            layer.weight.data = torch.ones(3, self.in_channels, 1, 1) / self.in_channels
        self.proj_to_rgb = layer.to(device=self.device, dtype=torch.bfloat16)

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)
        B, C, H, W = image_tensor.shape
        zH, zW = H // self.vae_compression, W // self.vae_compression
        
        enc_key = next((k for k in self.state_dict if "encoder.conv_in" in k or "encode_lq" in k), None)
        if enc_key:
            conv_w = self.state_dict[enc_key].to(self.device, dtype=torch.bfloat16)
            pad = conv_w.shape[-1] // 2
            x = F.conv2d(image_tensor, conv_w, stride=self.vae_compression, padding=pad)
        else:
            x = F.interpolate(image_tensor, size=(zH, zW), mode="bilinear", align_corners=False)
            x = torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)
            
        return x

    def decode(self, latent: torch.Tensor, caption: str = "", seed: int = 42) -> torch.Tensor:
        # Жёсткая синхронизация устройства и типа данных
        latent = latent.to(self.device, dtype=torch.bfloat16)
        
        B, C, zH, zW = latent.shape
        if C == 128 and self.in_channels == 16:
            latent = latent.view(B, 16, 8, 8, zH, zW).mean(dim=2).mean(dim=2)
        elif C != self.in_channels:
            raise ValueError(f"Latent channels ({C}) != expected ({self.in_channels})")
            
        H_out = zH * self.vae_compression * self.scale
        W_out = zW * self.vae_compression * self.scale
        
        # up_latent и proj_to_rgb гарантированно bfloat16
        up_latent = F.interpolate(latent, size=(H_out, W_out), mode="bicubic", align_corners=False)
        up_rgb = self.proj_to_rgb(up_latent)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn_like(up_rgb, generator=generator, dtype=torch.bfloat16)
        
        dec_in_key = next((k for k in self.state_dict if "decoder.conv_in" in k or "proj_out" in k), None)
        dec_out_key = next((k for k in self.state_dict if "decoder.conv_out" in k or "final_proj" in k), None)
        
        cond_strength = self.cfg_scale if caption.strip() else 0.0
        
        for step in range(self.inference_steps):
            t = step / max(self.inference_steps - 1, 1)
            
            if dec_in_key and dec_out_key:
                w_in = self.state_dict[dec_in_key].to(self.device, dtype=torch.bfloat16)
                w_out = self.state_dict[dec_out_key].to(self.device, dtype=torch.bfloat16)
                
                # Адаптация выходного канала под RGB, если в весах другое количество
                if w_out.shape[0] != 3:
                    w_out_adj = torch.zeros(3, w_out.shape[1], *w_out.shape[2:], device=self.device, dtype=torch.bfloat16)
                    w_out_adj[:min(3, w_out.shape[0])] = w_out[:min(3, w_out.shape[0])]
                    w_out = w_out_adj
                    
                h = F.conv2d(x, w_in, padding=w_in.shape[-1]//2)
                h = F.silu(h)
                h = F.conv2d(h, w_out, padding=w_out.shape[-1]//2)
                x = x + cond_strength * (h - x) * (1.0 - t)
            else:
                x = x * (1.0 - t) + up_rgb * t * (0.5 + 0.5 * cond_strength)
                
            if step < self.inference_steps - 1:
                noise = torch.randn_like(x, generator=generator, dtype=torch.bfloat16)
                x = x * 0.9 + noise * 0.1
                
        return torch.clamp(x, -1.0, 1.0)
