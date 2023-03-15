from diffusers import StableDiffusionPipeline
import torch

model_id = "./sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs"
image = pipe(prompt).images[0]  
    
image.save("pokemon.png")
