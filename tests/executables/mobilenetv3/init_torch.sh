bash ../_utils/init_classification_torch.sh ../_utils

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

command -v yum >/dev/null && $prefix_sudo yum install -y numactl ||  $prefix_sudo apt install -y numactl

pip3 install -r ../../../models/cv/classification/mobilenetv3/pytorch/requirements.txt