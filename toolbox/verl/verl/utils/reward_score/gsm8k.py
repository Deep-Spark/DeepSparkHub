# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    # 参考kk中的处理，模型的输出在关键字Assistant:之后，如果是qwen的话关键字是<|im_start|>assistant
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, None, False

    # this also tests the formatting of the model
    solution = re.search("####\s*([-+]?[^\d]*(\d+(?:\.\d+)?))", processed_str)
    if solution is None:
        format_correct = False
        final_answer = None
    else:
        format_correct = True
        final_answer = solution.group(2).replace(',', '')

    return final_answer, processed_str, format_correct


def compute_score(solution_str, ground_truth, method='strict', format_score=1., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Extract model answer
    answer, processed_str, format_correct = extract_solution(solution_str=solution_str, method=method)
    print(f"[Ground Truth] : {ground_truth}")
    print(f"\n[Model Response]\n{processed_str}")
    print(f"\n[extract_solution]\n{answer}")
    
    # Validate response structure
    format_score = format_score if format_correct else 0
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    if answer == ground_truth:
        answer_score = score
    else:
        answer_score = 0
    total_score = format_score + answer_score
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: ", format_score)
    print(f"  Answer: ", answer_score)
    print(f"  Total: ", total_score)
    return total_score