#/bin/bash
source $(cd `dirname $0`; pwd)/../_utils/which_install_tool.sh

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

if command_exists apt; then
	$prefix_sudo apt install -y git numactl
elif command_exists dnf; then
	$prefix_sudo dnf install -y git numactl
else
	$prefix_sudo yum install -y git numactl
fi
if [ "$(ulimit -n)" -lt "1048576" ]; then
	ulimit -n 1048576
fi
pip3 uninstall -y protobuf
pip3 install "protobuf<4.0.0"
pip3 install git+https://github.com/mlperf/logging.git
pip3 install git+https://github.com/NVIDIA/dllogger.git
pip3 install pandas==1.3.5
pip3 install numpy==1.26.4
