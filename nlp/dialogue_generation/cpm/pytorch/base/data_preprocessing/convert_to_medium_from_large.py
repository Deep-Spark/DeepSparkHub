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
import torch
import argparse
import collections

def convert_large_to_medium(sd):
    od = collections.OrderedDict()

    for para_name in sd['module']:
        if 'word_embeddings.weight' == para_name:
            od[para_name] = sd['module'][para_name][:,:1024].clone()
        if 'position_embeddings.weight' == para_name:
            od[para_name] = sd['module'][para_name][:,:1024].clone()

        for i in range(24):
            if f'transformer.layers.{i}.input_layernorm.weight' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break
            elif f'transformer.layers.{i}.input_layernorm.bias' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break
            elif f'transformer.layers.{i}.attention.query_key_value.weight' == para_name:              
                od[para_name] = sd['module'][para_name][:3072,:1024].clone()
                break

            elif f'transformer.layers.{i}.attention.query_key_value.bias' == para_name:
                od[para_name] = sd['module'][para_name][:3072].clone()
                break

            elif f'transformer.layers.{i}.attention.dense.weight' == para_name:
                od[para_name] = sd['module'][para_name][:1024,:1024].clone()
                break
            elif f'transformer.layers.{i}.attention.dense.bias' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break
            elif f'transformer.layers.{i}.post_attention_layernorm.weight' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break
            elif f'transformer.layers.{i}.post_attention_layernorm.bias' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break
            elif f'transformer.layers.{i}.mlp.dense_h_to_4h.weight' == para_name:
                od[para_name] = sd['module'][para_name][:4096,:1024].clone()
                break
            elif f'transformer.layers.{i}.mlp.dense_h_to_4h.bias' == para_name:
                od[para_name] = sd['module'][para_name][:4096].clone()
                break
            elif f'transformer.layers.{i}.mlp.dense_4h_to_h.weight' == para_name:
                od[para_name] = sd['module'][para_name][:1024,:4096].clone()
                break
            elif f'transformer.layers.{i}.mlp.dense_4h_to_h.bias' == para_name:
                od[para_name] = sd['module'][para_name][:1024].clone()
                break

        if 'transformer.final_layernorm.weight' == para_name:
            od[para_name] = sd['module'][para_name][:1024].clone()
        elif 'transformer.final_layernorm.bias' == para_name:
            od[para_name] = sd['module'][para_name][:1024].clone()
        else:
            pass

    return od

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--load", default="", type=str)
    args = arg_parser.parse_args()

    sd = torch.load(args.load, map_location='cpu')
    # print(f"sd:")
    # for item_name in sd:
    #     print(item_name)

    # print(f"sd_model:")
    # for para_name in sd['module']:
    #     print(para_name, sd['module'][para_name].size())

    od = convert_large_to_medium(sd)

    # print(f"od:")
    # for para_name in od:
    #     print(para_name,od[para_name].size())

    medium_file = args.load[:args.load.rfind('/')+1]+'cpm_model_states_medium.pt'
    torch.save(od, medium_file)