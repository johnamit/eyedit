import os
import sys
import torch
import json
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL

# Path Setup
# add SiT folder to python path so we can impot models and transport
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sit_module_path = os.path.join(project_root, "SiT")

if sit_module_path not in sys.path:
    sys.path.append(sit_module_path)
    
# Import from SiT
try:
    from models import SiT_models
    from transport import create_transport, Sampler
except ImportError as e:
    raise ImportError(f"Could not import SiT modules. Ensure '{sit_module_path}' contains models.py and transport.py. Error: {e}")


# Storing the loaded model
class SiTGenerator:
    """A class that holds the loaded model in memory for inference, instead of reloading it each time."""
    def __init__(self, ckpt_path, mapping_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_path = ckpt_path
        self.mapping_path = mapping_path
        
        # Load mapping file
        self._load_mappings()
        
        # Load checkpoint
        print(f"Loading checkpoint: {self.ckpt_path}")
        self.checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        
        if "args" in self.checkpoint:
            self.args = self.checkpoint["args"]
        else:
            raise ValueError("Checkpoint does not contain 'args'. Cannot determine model architecture.")

        # Initialise and load model
        self._init_model()
        
        # initialise VAE and transport
        self._init_transport()
        
        print("SiT Generator initialized and ready for inference.") 


    def _load_mappings(self):
            with open(self.mapping_path, 'r') as f:
                self.gene_to_idx = json.load(f)
    
               
    def _init_model(self):
        # Calculate latent size (e.g., 256 // 8 = 32)
        self.latent_size = self.args.image_size // 8
        
        # Create Model Architecture
        self.model = SiT_models[self.args.model](
            input_size=self.latent_size,
            num_classes=self.args.num_classes
        ).to(self.device)
        
        # Load Weights (EMA preferred)
        if "ema" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["ema"])
        else:
            self.model.load_state_dict(self.checkpoint["model"])
        
        self.model.eval()  # Set to inference mode
    
      
    def _init_transport(self):
        # Load VAE (Standard Stable Diffusion Autoencoder)
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}").to(self.device)
        
        # Setup Diffusion Transport
        self.transport = create_transport(
            self.args.path_type,
            self.args.prediction,
            self.args.loss_weight,
            self.args.train_eps,
            self.args.sample_eps
        )
        self.sampler = Sampler(self.transport)
        
    @torch.no_grad()
    def generate(self, gene, laterality, age, cfg_scale=4.0):
        """Generates a single image based on parameters. Returns a PIL Image."""
        
        ## Pre-processing
        # Gene
        if gene not in self.gene_to_idx:
            raise ValueError(f"Gene '{gene}' not found in mapping.")
        gene_idx = self.gene_to_idx[gene]
        
        # Laterality
        lat_map = {'L': 0, 'Left': 0, 'left': 0, 'R': 1, 'Right': 1, 'right': 1}
        lat_idx = lat_map.get(laterality, 1)  # Default to R if unknown
        
        # Age Normalization
        age_norm = float(age) / 100.0
        age_norm = max(0.0, min(1.0, age_norm))
        
        ## Tensor Preparation
        n = 1  # Single sample
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=self.device)
        
        model_kwargs = dict(
            genes=torch.tensor([gene_idx] * n, device=self.device),  # Plural 'genes'
            lats=torch.tensor([lat_idx] * n, device=self.device),    # 'lats', not 'laterality'
            ages=torch.tensor([age_norm] * n, device=self.device)    # Plural 'ages'
        )
        
       ## Classifier-Free Guidance Setup
        if cfg_scale > 1.0: 
            # Create null tokens for classifier-free guidance
            z = torch.cat([z, z], 0)
            model_kwargs['genes'] = torch.cat([model_kwargs['genes'], torch.tensor([self.args.num_classes] * n, device=self.device)], 0)
            model_kwargs['lats'] = torch.cat([model_kwargs['lats'], torch.tensor([2] * n, device=self.device)], 0)
            model_kwargs['ages'] = torch.cat([model_kwargs['ages'], torch.tensor([-1.0] * n, device=self.device)], 0)
            model_kwargs['cfg_scale'] = cfg_scale
            model_fn = self.model.forward_with_cfg
        else:
            model_fn = self.model.forward
            
        # Sampling
        samples = self.sampler.sample_ode()(z, model_fn, **model_kwargs)[-1] # Get final sample
        
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # Keep only the first half
            
        # Decode with VAE
        samples = self.vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # convert to PIL Image
        return Image.fromarray(samples[0])

    @torch.no_grad()
    def invert(self, pil_image, gene, laterality, age,
               sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3):
        """
        Invert a PIL image to its latent noise representation using ODE inversion.
        
        Args:
            pil_image: PIL Image to invert
            gene: Gene symbol string (e.g. 'ABCA4')
            laterality: 'L' or 'R'
            age: Patient age (int)
            
        Returns:
            z_noise: Inverted noise tensor (kept on device)
        """
        # --- Pre-processing ---
        if gene not in self.gene_to_idx:
            raise ValueError(f"Gene '{gene}' not found in mapping.")
        gene_idx = self.gene_to_idx[gene]
        
        lat_map = {'L': 0, 'Left': 0, 'left': 0, 'R': 1, 'Right': 1, 'right': 1}
        lat_idx = lat_map.get(laterality, 1)
        
        age_norm = float(age) / 100.0
        age_norm = max(0.0, min(1.0, age_norm))
        
        # --- Encode image to latent space via VAE ---
        img = pil_image.convert("RGB").resize(
            (self.args.image_size, self.args.image_size), Image.BICUBIC
        )
        x = np.array(img).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        z_data = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        
        # --- Prepare conditioning ---
        model_kwargs = dict(
            genes=torch.tensor([gene_idx], device=self.device),
            lats=torch.tensor([lat_idx], device=self.device),
            ages=torch.tensor([age_norm], device=self.device),
        )
        
        # --- ODE Backward: Data(t=1) -> Noise(t=0) ---
        inverse_ode_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            reverse=True,  # data -> noise
        )
        z_noise = inverse_ode_fn(z_data, self.model.forward, **model_kwargs)[-1]
        return z_noise

    @torch.no_grad()
    def edit(self, z_noise, target_gene, target_laterality, target_age,
             sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3):
        """
        Generate an edited image from inverted noise with new conditioning.
        
        Args:
            z_noise: Inverted noise tensor from invert()
            target_gene: Target gene symbol string
            target_laterality: Target laterality ('L' or 'R')
            target_age: Target age (int)
            
        Returns:
            PIL Image of the edited result
        """
        # --- Pre-processing ---
        if target_gene not in self.gene_to_idx:
            raise ValueError(f"Gene '{target_gene}' not found in mapping.")
        gene_idx = self.gene_to_idx[target_gene]
        
        lat_map = {'L': 0, 'Left': 0, 'left': 0, 'R': 1, 'Right': 1, 'right': 1}
        lat_idx = lat_map.get(target_laterality, 1)
        
        age_norm = float(target_age) / 100.0
        age_norm = max(0.0, min(1.0, age_norm))
        
        # --- Prepare conditioning ---
        model_kwargs = dict(
            genes=torch.tensor([gene_idx], device=self.device),
            lats=torch.tensor([lat_idx], device=self.device),
            ages=torch.tensor([age_norm], device=self.device),
        )
        
        # --- ODE Forward: Noise(t=0) -> Data(t=1) ---
        forward_ode_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            reverse=False,
        )
        z_edited = forward_ode_fn(z_noise, self.model.forward, **model_kwargs)[-1]
        
        # --- Decode to image ---
        edited_img = self.vae.decode(z_edited / 0.18215).sample
        edited_img = torch.clamp(127.5 * edited_img + 128.0, 0, 255)
        edited_img = edited_img.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        
        return Image.fromarray(edited_img)