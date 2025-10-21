pip3 install -r requirements.txt
pip3 install -e mlperf-logging

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y numactl
else
    yum install -y numactl
fi
