set -euox pipefail

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y numactl 
else
    yum install -y numactl
fi

pip3 install -r requirements.txt

