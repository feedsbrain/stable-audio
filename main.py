import torch
import torchaudio

from uuid import uuid4
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# output path
file_path = f"./outputs/{uuid4()}.wav"


# get torch device
def get_torch_device():
  # Check available devices
  if torch.cuda.is_available():
      device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  return device

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

device = get_torch_device()
model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "128bpm techno drum beat loop",
    "seconds_start": 0,
    "seconds_total": 30
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save(file_path, output, sample_rate)
