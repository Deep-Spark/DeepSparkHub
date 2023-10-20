# ControlNet

## Model description

ControlNet is a neural network structure to control diffusion models by adding extra conditions, a game changer for AI Image generation. It brings unprecedented levels of control to Stable Diffusion. The revolutionary thing about ControlNet is its solution to the problem of spatial consistency.

This is simple: we want to control SD to fill a circle with colors, and the prompt contains some description of our target.

Stable diffusion is trained on billions of images, and it already knows what is "cyan", what is "circle", what is "pink", and what is "background".

But it does not know the meaning of that "Control Image (Source Image)". Our target is to let it know.

## Step 1: Installation

- Install
    ```bash
    pip3 install open_clip_torch
    pip3 install transformers
    pip3 install einops
    pip3 install omegaconf
    pip3 install pytorch-lightning==1.9.5

    ```
- Build the Stable Difussion to control
    You need to decide which Stable Diffusion Model you want to control. In this example, we will just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). (Or ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) if you are using SD2.)
    ```bash
    # We provide a simple script for you to achieve this easily. 
    
    #If your SD filename is "./models/v1-5-pruned.ckpt" and you want the script to save the processed model (SD+ControlNet) at location "./models/control_sd15_ini.ckpt", you can just run:

    python3 tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt

    # Or if you are using SD2:

    python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
    ```

## Step 2: Preparing datasets

Just download the Fill50K dataset from [our huggingface page](https://huggingface.co/lllyasviel/ControlNet) (training/fill50k.zip, the file is only 200M!). Make sure that the data is decompressed as 

    ControlNet/training/fill50k/prompt.json
    ControlNet/training/fill50k/source/X.png
    ControlNet/training/fill50k/target/X.png

In the folder "fill50k/source", you will have 50k images of circle lines.

In the folder "fill50k/target", you will have 50k images of filled circles.

In the "fill50k/prompt.json", you will have their filenames and prompts. Each prompt is like "a balabala color circle in some other color background."

## Step 3: Training

```bash
# One GPU
python3 tutorial_train.py

# 8 GPUs
python3 tutorial_train_dist.py

```
