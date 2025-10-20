#!/bin/bash
# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

PIPCMD=pip3
: ${PKGS_CACHE_DIR:="__null__"}

function install_pip_pkgs() {
    for pkg in "$@"
    do
        if [ ! -d $PKGS_CACHE_DIR ]; then
            $PIPCMD install $pkg
        else
            $PIPCMD install --no-index --find-links=$PKGS_CACHE_DIR $pkg
        fi
    done
}

# Exeample
# pkgs=(1 2 3)
# install_pip_pkgs "${pkgs[@]}"