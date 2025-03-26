# HashNeRF

## Model description

HashNeRF is an efficient implementation of Neural Radiance Fields (NeRF) using a multiresolution hash encoding
technique. It accelerates 3D scene reconstruction and novel view synthesis by optimizing memory usage and computational
efficiency. Based on instant-ngp's approach, HashNeRF employs a grid encoder and density grid ray sampler to achieve
high-quality rendering results. The model supports various datasets and custom scenes, making it suitable for
applications in computer graphics, virtual reality, and 3D reconstruction tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

We use the same data format as instant-ngp, [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox) and
blender dataset [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).Please
download and put them under `./data`.

For custom dataset, you should:

1. take a video / many photos from different views
2. put the video under a path like ./data/custom/video.mp4 or the images under ./data/custom/images/*.jpg.
3. call the preprocess code: (should install ffmpeg and colmap first! refer to the file for more options)

```bash
python3 scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python3 scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

First time running will take some time to compile the CUDA extensions.

```bash
# train with fox dataset on One single GPU
python3 main_nerf.py data/fox --workspace trial_nerf -O

# data/fox is dataset path; --workspace means output path;
# -O means --fp16 --cuda_ray --preload, which usually gives the best results balanced on speed & performance.

# test mode
python3 main_nerf.py data/fox --workspace trial_nerf -O --test

# train with the blender dataset, you should add `--bound 1.0 --scale 0.8 --dt_gamma 0`
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box. 
# --dt_gamma controls the adaptive ray marching speed, set to 0 turns it off.
python3 main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --bound 1.0 --scale 0.8 --dt_gamma 0

# train with custom dataset(you'll need to tune the scale & bound if necessary):
python3 main_nerf.py data/custom_data --workspace trial_nerf -O
```

## Model Results

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.0652               | SDK V2.2,bs:1,1x,fp16                    | 10          | 11.9     | 82         | 0.903       | 28.1                    | 1         |

## Reference

- [torch-ngp](https://github.com/ashawkey/torch-ngp)
- [DearPyGui](https://github.com/hoffstadt/DearPyGui)
