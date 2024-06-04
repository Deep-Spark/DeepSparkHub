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

start_time=$(date +%s)
unset no_proxy use_proxy https_proxy http_proxy
EXIT_STATUS=0
check_status() {
	if ((${PIPESTATUS[0]} != 0)); then
		EXIT_STATUS=1
	fi
}

python3 -m torch.distributed.launch --nproc_per_node=16 \
	train.py --batch-size 32 \
	--data ./data/coco.yaml --weights "" \
	--cfg models/yolov5m.yaml --workers 16 \
	--epochs 3 --linear-lr "$@"
check_status

wait

end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))
echo "end to end time: $e2e_time" >>total_time.log
exit ${EXIT_STATUS}
