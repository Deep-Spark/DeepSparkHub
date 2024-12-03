# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from kan import *
from kan.utils import create_dataset
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help= "training epoch")
    parser.add_argument("--batch_size", type=int, default=1000, help= "training batch size")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)


    # create dataset f(x,y) = exp(sin(pi*x)+y^2)
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, device=device)
    dataset['train_input'].shape, dataset['train_label'].shape

    model(dataset['train_input'])

    t_start = time.time()

    model.fit(dataset, opt="LBFGS", steps=args.steps, lamb=0.001, batch=args.batch_size)

    t_elapse = time.time() - t_start
    real_batchz_size = args.batch_size if args.batch_size < dataset['train_input'].shape[0] else dataset['train_input'].shape[0]
    total_samples = args.steps * real_batchz_size
    print(f"finished {total_samples} samples, total time: {t_elapse:5.3} s, samples_per_second:{total_samples/t_elapse:7.3}") 


if __name__ == "__main__":
    main()