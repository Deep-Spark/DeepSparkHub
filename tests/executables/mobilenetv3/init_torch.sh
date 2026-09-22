bash ../_utils/init_classification_torch.sh ../_utils

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

if command -v yum >/dev/null; then
    $prefix_sudo yum install -y numactl
else
    export DEBIAN_FRONTEND=noninteractive
    $prefix_sudo apt-get update -qq && $prefix_sudo apt-get install -y --no-install-recommends numactl || exit 1
fi

pip3 install -r ../../../models/cv/classification/mobilenetv3/pytorch/requirements.txt