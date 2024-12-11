# Copyright 2019 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
with open("VERSION", "r") as f:
    version = f.read().strip()

setuptools.setup(
    name="mlperf-logging",
    version=version,
    author="MLPerf.org",
    author_email="mlperf@googlegroups.com",
    description="MLPerf compliance tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlperf/logging",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    include_package_data=True,
    install_requires=[
        'pandas>=1.0', 'pyyaml>=5.4.1', 'numpy>=1.17.3', 'scipy>=1.4.1'
    ],
)
