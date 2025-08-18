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

from importlib.metadata import version, PackageNotFoundError
from packaging import version as vs


def get_version(pkg):
    try:
        ver = version(pkg)
        if "+" in ver:
            return ver.split("+")[0]
        else:
            return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)
vllm_version = None

if package_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
elif vs.parse(package_version) >= vs.parse('0.6.6.post2.dev252+g8027a724'):
    # From 0.6.6.post2 on, vllm supports SPMD inference
    # See https://github.com/vllm-project/vllm/pull/12071

    from vllm import LLM
    from vllm.distributed import parallel_state
    from .vllm_spmd.dtensor_weight_loaders import load_dtensor_weights
else:
    raise ValueError(
        f'vllm version {package_version} not supported. Currently supported versions are 0.3.1, 0.4.2, 0.5.4, 0.6.3 and 0.7.0+'
    )
