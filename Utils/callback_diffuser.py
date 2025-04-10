import os
import torch
from PIL import Image
import glob
import time
import threading


# convert latent image to RGB for display 
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)

def create_callback_on_step_end(PREVIEW_QUEUE,stop_gen):  
    def callback_on_step_end(pipe, step, timestep, callback_kwargs):
        if stop_gen.is_set():
            pipe._interrupt = True
        latents = callback_kwargs["latents"]
        image = latents_to_rgb(latents[0])
        PREVIEW_QUEUE.append(image)
        return callback_kwargs
    return callback_on_step_end

def interrupt_diffusers_callback(stop_gen):
    def interrupt_callback(pipe, step, timestep, callback_kwargs):
        if stop_gen.is_set():
            pipe._interrupt = True
        return callback_kwargs
    return interrupt_callback