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
import shutil
import subprocess
from typing import List, Optional, Union
import os.path as ospath
from argparse import ArgumentParser, REMAINDER
from functools import partial, wraps
from typing import NamedTuple


# =========================================================
# Define arguments
# =========================================================

def parse_args():
    parser = ArgumentParser("Prepare")
    parser.add_argument("--name", type=str, default=None, help="The name of submitter")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    # WARN: Don't delete this argument
    parser.add_argument('other_args', nargs=REMAINDER)
    args = parser.parse_args()
    return args


# =========================================================
# Constants
# =========================================================
args              = parse_args()

APT               = "apt"
PIP_INSTALL       = "pip3 install "
PYTHON            = "python3"
DOWNLOAD          = "wget"

APT_PKGS          = ["numactl"]
SUPPORTED_WHEELS  = ["torch", "apex"]


MODEL_DIR = ospath.abspath(
    ospath.join(
        __file__,
        "../../"
    )
)
CURRENT_MODEL_NAME = ospath.basename(MODEL_DIR)
PROJ_DIR = ospath.abspath(
    ospath.join(
        MODEL_DIR,
        "../../"
    )
)

PACKAGE_DIR_NAME  = "packages"
SOURCE_DIR_NAME   = "csrc"
SDK_DIR_NAME      = "sdk_installers"
PACKAGE_LIST_NAME = "files.txt"
SDK_LIST_NAME     = "files.txt"

DATA_SHARDS       = 2048
TRAIN_DIR_NAME    = f"{DATA_SHARDS}_shards_uncompressed"
CONVERTED_CHECKPOINT_NAME = "model.ckpt-28252.pt"

SUBMITTER         = args.name
DATA_DIR          = args.data_dir
SUBMITTER_DIR     = ospath.join(PROJ_DIR, SUBMITTER, CURRENT_MODEL_NAME)
TRAIN_DATA_DIR    = ospath.join(DATA_DIR, TRAIN_DIR_NAME)

EXTENSION_SOURCE_DIR_ENV = "EXTENSION_SOURCE_DIR"
SDK_ARGUMENTS_ENV        = "SDK_ARGUMENTS"


# =========================================================
# Helpers
# =========================================================

class ShellResult(NamedTuple):

    returncode: int
    result: Union[subprocess.CompletedProcess, str] = None


def _exec_cmd(cmd: Union[str, List], *args, **kwargs):
    args_str = " ".join(args)
    args_str += " ".join([f"{name}={value}" for name, value in kwargs.items()])
    cmd_str = cmd
    if isinstance(cmd, (tuple, list)):
        cmd_str = " ".join(cmd)
    print(f"Commands: {cmd_str}")

    result = subprocess.run(cmd, *args, **kwargs, stdout=None, stderr=subprocess.STDOUT)

    if result.returncode > 0:
        msg = f"ERROR: {cmd} {args_str}"
        return ShellResult(returncode=result.returncode, result=msg)

    return ShellResult(returncode=result.returncode, result=result)


def exec_shell_cmd(cmd: str, *args, **kwargs):
    return _exec_cmd(cmd, shell=True, *args, **kwargs)


def exec_shell_cmds(cmds: List[str], *args, **kwargs):
    cmds = "\n".join(cmds)
    return exec_shell_cmd(cmds, *args, **kwargs)


def warning(*args, **kwargs):
    print("WARN:", *args, **kwargs)


def find_file_by_match(dir, file_patterns):
    if ospath.exists(dir):
        dir_files = os.listdir(dir)
    else:
        return file_patterns

    for i, pattern in enumerate(file_patterns):
        pattern = pattern.strip()
        if len(pattern) <= 1 or not pattern.endswith("*"):
            continue

        pattern = pattern[:-1]

        for dir_file in dir_files:
            if dir_file.startswith(pattern):
                file_patterns[i] = dir_file
                break
    return file_patterns

# =========================================================
# Pipelines
# =========================================================

def install_apt_packages():
    if len(APT_PKGS) == 0:
        return
    return exec_shell_cmd(f"{APT} install -y {' '.join(APT_PKGS)}")


def prepare_data():
    checked_files = ["2048_shards_uncompressed", "bert_config.json", "eval_set_uncompressed", "model.ckpt-28252.pt"]
    path_join = ospath.join
    exist_preprocessed_data = all([ospath.exists(path_join(DATA_DIR, name)) for name in checked_files])

    if exist_preprocessed_data:
        return

    # Check last download file
    origin_data = path_join(DATA_DIR, "phase1", "model.ckpt-28252.meta")
    need_download_dataset = "-s" if ospath.exists(origin_data) else ""
    cmds = [
        f"cd {path_join(MODEL_DIR, 'pytorch/data_preprocessing')}",
        f"bash prepare_data.sh -o {DATA_DIR} -p {DATA_SHARDS} {need_download_dataset}"
    ]
    return exec_shell_cmds(cmds)


def install_sdk():
    def get_sdk_args():
        sdk_args = dict()
        if SDK_ARGUMENTS_ENV in os.environ:
            sdk_args_str = os.environ[SDK_ARGUMENTS_ENV]

            sdk_args_segments = sdk_args_str.split(';')
            for sdk_arg in sdk_args_segments:
                sdk, arg = sdk_arg.split('=', maxsplit=1)
                sdk_args[sdk] = arg
        return sdk_args

    sdk_args_dict = get_sdk_args()
    print("SDK Arguments:", sdk_args_dict)

    sdk_installer_dir = ospath.join(SUBMITTER_DIR, SDK_DIR_NAME)
    if not ospath.exists(sdk_installer_dir):
        sdk_installer_dir = ospath.join(PROJ_DIR, SUBMITTER, SDK_DIR_NAME)
        if not ospath.exists(sdk_installer_dir):
            warning("Not found sdk\'s dir, skip run installer")
            return

    # Find sdk installers
    sdk_list_file = ospath.join(sdk_installer_dir, SDK_LIST_NAME)
    if ospath.exists(sdk_list_file):
        with open(sdk_list_file) as f:
            sdk_installers = f.readlines()
        sdk_installers_pattern = [sdk.strip() for sdk in sdk_installers]
        sdk_installers = find_file_by_match(sdk_installer_dir, sdk_installers_pattern)
    else:
        sdk_installers = os.listdir(sdk_installer_dir)
        sdk_installers.sort()

    sdk_installers_cmds = []
    for sdk in sdk_installers:
        if sdk.endswith(".run"):
            sdk_arg = ""
            for sdk_args_key in sdk_args_dict:
                if sdk.startswith(sdk_args_key):
                    sdk_arg = sdk_args_dict[sdk_args_key]
            sdk_installers_cmds.append("sh " + ospath.join(sdk_installer_dir, sdk) + f" {sdk_arg}")

    if len(sdk_installers_cmds) == 0:
        warning("Not found installer in", sdk_installer_dir, ", skip run installer")
        return

    return exec_shell_cmds(sdk_installers_cmds)


def install_requirements():
    return exec_shell_cmd(
        f"{PIP_INSTALL} -r requirements.txt"
    )


def install_wheel_pkgs(filter_packages: bool=False):
    wheel_dir = ospath.join(SUBMITTER_DIR, PACKAGE_DIR_NAME)
    if not ospath.exists(wheel_dir):
        warning("Not found package\'s dir, skip install wheel package")
        return

    # Find packages
    package_list_file = ospath.join(wheel_dir, PACKAGE_LIST_NAME)
    if ospath.exists(package_list_file):
        with open(package_list_file) as f:
            packages = f.readlines()
        packages_pattern = [pkg.strip() for pkg in packages]
        packages = find_file_by_match(wheel_dir, packages_pattern)
    else:
        packages = os.listdir(wheel_dir)
        packages.sort()

    def _filter_packages(name: str):
        for support_pkg in SUPPORTED_WHEELS:
            if name.startswith(support_pkg):
                return True
        return False

    if filter_packages:
        packages = list(filter(_filter_packages, packages))

    if len(packages) == 0:
        warning("Not found wheel packages in", wheel_dir)
        return

    install_packages_cmds = [f"{PIP_INSTALL} {ospath.join(wheel_dir, pkg)}" for pkg in packages]
    return exec_shell_cmds(install_packages_cmds)


def install_extensions():
    source_dir = ospath.join(SUBMITTER_DIR, SOURCE_DIR_NAME)
    if not ospath.exists(source_dir):
        warning("Not found source dir:", source_dir)
        return

    sandbox_dir = os.path.join(MODEL_DIR, "pytorch", 'sandbox', "extension")
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)

    cmds = [
        f"export {EXTENSION_SOURCE_DIR_ENV}={source_dir}",
        f"mkdir -p {sandbox_dir}",
        f"cd {sandbox_dir}",
        f"{PYTHON} ../../setup.py install",
        f"rm -rf {sandbox_dir}"
    ]

    return exec_shell_cmds(cmds)


def pipelines():
    return [
        # TODO: Uncomment
        install_apt_packages,
        install_requirements,
        install_sdk,
        partial(install_wheel_pkgs, filter_packages=True),
        install_extensions,
        # prepare_data,
    ]


if __name__ == '__main__':
    for pipeline in pipelines():
        result = pipeline()
        if result is not None and result.returncode > 0:
            print(result.result)
            print("Fail:", pipeline)
            exit(result.returncode)





