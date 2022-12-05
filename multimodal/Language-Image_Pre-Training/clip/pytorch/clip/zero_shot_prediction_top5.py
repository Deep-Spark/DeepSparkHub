# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import os
import clip
import torch
from torchvision.datasets import CIFAR100
import time
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

torch.cuda.synchronize()
start = time.time()
true_classify = 0
with torch.no_grad():
    # Prepare the inputs
    for i in tqdm(range(10000)):
        #print("这是第%d张图片" %i)
        image, class_id = cifar100[i]
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        #print(class_id)
        # Calculate features
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        #print(values)
        #print(indices)
        if class_id in indices:
            true_classify += 1
    #print(true_classify)

#计算总用时
torch.cuda.synchronize()
end = time.time()
total_time = end-start
fps = 10000/total_time
print("fps:%f" %(fps))


accuracy = true_classify/10000
print("accuracy:%f" %accuracy)

