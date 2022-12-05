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
# Install packages
. install_pip_pkgs.sh

pkgs=('transformers==4.12.5' 'sacrebleu==2.0.0' 'datasets==1.16.1')
install_pip_pkgs "${pkgs[@]}"


# Check transformers
if python3 -c "import transformers" >/dev/null 2>&1
then
    echo "Transformers already installed"
else
    echo "Again try to install transformers package"
    source install_rust.sh
    if [ ! -d "./packages" ]; then
        $PIPCMD install --ignore-installed PyYAML
        $PIPCMD install transformers==4.12.5
    else
        $PIPCMD install --ignore-installed --no-index --find-links=./packages PyYAML
        $PIPCMD install --no-index --find-links=./packages transformers
    fi
fi