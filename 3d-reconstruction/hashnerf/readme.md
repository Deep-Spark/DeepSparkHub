# torch-ngp
## Model description
A pytorch implementation of the NeRF part (grid encoder, density grid ray sampler) in instant-ngp, as described in Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```

## Step 2: Prepare dataset

We use the same data format as instant-ngp, [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox) and blender dataset [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).Please download and put them under `./data`.

For custom dataset, you should:
1. take a video / many photos from different views 
2. put the video under a path like ./data/custom/video.mp4 or the images under 
./data/custom/images/*.jpg.
3. call the preprocess code: (should install ffmpeg and colmap first! refer to the file for more options)
```bash
python scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
```

## Step 3: Training and test

### One single GPU

First time running will take some time to compile the CUDA extensions.

```bash
# train with fox dataset
python main_nerf.py data/fox --workspace trial_nerf -O
# data/fox is dataset path; --workspace means output path; -O means --fp16 --cuda_ray --preload, which usually gives the best results balanced on speed & performance.

# test mode
python main_nerf.py data/fox --workspace trial_nerf -O --test
```

```bash
# train with the blender dataset, you should add `--bound 1.0 --scale 0.8 --dt_gamma 0`
# --bound means the scene is assumed to be inside box[-bound, bound]
# --scale adjusts the camera locaction to make sure it falls inside the above bounding box. 
# --dt_gamma controls the adaptive ray marching speed, set to 0 turns it off.
python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O --bound 1.0 --scale 0.8 --dt_gamma 0
```

```bash
# train with custom dataset(you'll need to tune the scale & bound if necessary):
python main_nerf.py data/custom_data --workspace trial_nerf -O
```

## Results on BI-V100

@@ -65,90 +60,4 @@ python main_nerf.py data/nerf_synthetic/lego --workspace trial_nerf -O


| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.0652               | SDK V2.2,bs:1,1x,fp16                    | 10          | 11.9     | 82         | 0.903       | 28.1                    | 1         |



## Reference

**Q**: How to choose the network backbone? 

**A**: The `-O` flag which uses pytorch's native mixed precision is suitable for most cases. I don't find very significant improvement for `--tcnn` and `--ff`, and they require extra building. Also, some new features may only be available for the default `-O` mode.

**Q**: CUDA Out Of Memory for my dataset.

**A**: You could try to turn off `--preload` which loads all images in to GPU for acceleration (if use `-O`, change it to `--fp16 --cuda_ray`). Another solution is to manually set `downscale` in `NeRFDataset` to lower the image resolution.

**Q**: How to adjust `bound` and `scale`? 

**A**: You could start with a large `bound` (e.g., 16) or a small `scale` (e.g., 0.3) to make sure the object falls into the bounding box. The GUI mode can be used to interactively shrink the `bound` to find the suitable value. Uncommenting [this line](https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py#L219) will visualize the camera poses, and some good examples can be found in [this issue](https://github.com/ashawkey/torch-ngp/issues/59).

**Q**: Noisy novel views for realistic datasets.

**A**: You could try setting `bg_radius` to a large value, e.g., 32. It trains an extra environment map to model the background in realistic photos. A larger `bound` will also help.


More information ref:ttps://github.com/ashawkey/torch-ngp