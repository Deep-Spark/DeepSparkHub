set -euox pipefail

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd $current_path

bash run_training.sh --name nvidia --config V100sx1x8 --data_dir /home/data/perf/glm