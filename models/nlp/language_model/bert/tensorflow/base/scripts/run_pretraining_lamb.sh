#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Changes:
# - Removed NVidia container build version message
###############################################################################

train_batch_size_phase2=${1:-10}
eval_batch_size=${2:-48}
learning_rate_phase2=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"false"}
num_hpus=${6:-8}
warmup_steps_phase2=${7:-"0"}
train_steps_phase2=${8:-8103}
save_checkpoints_steps=${9:-1000}
num_accumulation_steps_phase2=${10:-4}
bert_model=${11:-"large"}
is_dist_eval_enabled=${12:-"false"}
eval_only=${13:-"false"}

GBS2=$(expr $train_batch_size_phase2 \* $num_hpus \* $num_accumulation_steps_phase2)
printf -v TAG "tf_bert_pretraining_lamb_%s_%s_gbs2%d" "$bert_model" "$precision"  $GBS2
DATESTAMP=`date +'%y%m%d%H%M%S'`


samples_between_eval=150080
stop_threshold="0.72"
samples_start_eval=0
epochs=$(echo "( $train_steps_phase2 * $GBS2 ) / $samples_between_eval" | bc -l)
max_eval_steps=$(echo " 10000 / $eval_batch_size" | bc -l)
printf "Number of epochs: %.3f" "$epochs"

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

#printf -v SCRIPT_ARGS "%d %d %e %s %s %d %d %d %d %d %s %d %e %d %d %s %s" \
#                      $train_batch_size_phase2 $eval_batch_size  \
#                      $learning_rate_phase2 "$precision" "$use_xla" $num_hpus  \
#                      $warmup_steps_phase2  $train_steps_phase2 $save_checkpoints_steps \
#                      $num_accumulation_steps_phase2 "$bert_model" $samples_between_eval $stop_threshold $samples_start_eval $max_eval_steps "$is_dist_eval_enabled" "$eval_only"

echo "learning_rate_phase2=$learning_rate_phase2, warmup_steps_phase2=$warmup_steps_phase2, train_steps_phase2=$train_steps_phase2, num_accumulation_steps_phase2=$num_accumulation_steps_phase2, train_batch_size_phase2=$train_batch_size_phase2"
# RUN PHASE 2
#source ./TensorFlow/nlp/bert/scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
source ./scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
