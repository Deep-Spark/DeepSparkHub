pip3 install -r requirements.txt
pip3 install -e mlperf-logging

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y --no-install-recommends numactl || exit 1
else
    yum install -y numactl
fi
