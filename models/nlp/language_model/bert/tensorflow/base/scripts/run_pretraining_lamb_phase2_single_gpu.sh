#! /bin/bash

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

train_batch_size_phase2=${1:-7}
eval_batch_size=${2:-24}
learning_rate_phase2=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"false"}
num_hpus=${6:-8}
warmup_steps_phase2=${7:-"0"}
train_steps_phase2=${8:-8103}
save_checkpoints_steps=${9:-1000}
num_accumulation_steps_phase2=${10:-4}
bert_model=${11:-"large"}
samples_between_eval=${12:-150000}
stop_threshold=${13:-"0.720"}
samples_start_eval=${14:-3000000}
max_eval_steps=${15:-100}
is_dist_eval_enabled=${16:-"false"}
eval_only=${17:-"false"}


GBS2=$(expr $train_batch_size_phase2 \* $num_hpus \* $num_accumulation_steps_phase2)
printf -v TAG "tf_bert_pretraining_lamb_%s_%s_gbs2%d" "$bert_model" "$precision"  $GBS2
DATESTAMP=`date +'%y%m%d%H%M%S'`

epochs=$(echo "( $train_steps_phase2 * $GBS2 ) / $samples_between_eval" | bc -l)

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-./results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

#Edit to save logs & checkpoints in a different directory
#RESULTS_DIR=${RESULTS_DIR:-/results}

export BERT_CONFIG=./pretrain_ckpt/bert_config.json
export TF_FORCE_GPU_ALLOW_GROWTH="true"

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--amp"
elif [ "$precision" = "fp32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "tf32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--noamp --manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
   PREC="$PREC --use_xla"
   echo "XLA activated"
else
   PREC="$PREC --nouse_xla"
fi

#horovod_str="--horovod"
horovod_str=""

#PHASE 1 Config
PHASE1_CKPT=${INITIAL_CHECKPOINT:-./pretrain_ckpt}
PHASE1_CKPT=${PHASE1_CKPT}/model.ckpt-28252

#PHASE 2

seq_len=512 #$P2_MAX_SEQ_LENGTH
max_pred_per_seq=76 #####
gbs_phase2=$(expr $train_batch_size_phase2 \* $num_accumulation_steps_phase2)


#RESULTS_DIR_PHASE2=/results/tf_bert_pretraining_lamb_large_fp16_gbs2320_211231030404/phase_2
RESULTS_DIR_PHASE2=${RESULTS_DIR}/phase_2
mkdir -m 777 -p $RESULTS_DIR_PHASE2

INPUT_FILES=${INPUT_FILES:-/datasets/bert_tfrecord/train_data/}
EVAL_FILES=${EVAL_FILES:-/datasets/bert_tfrecord/eval_data/}

function check_dirs()
{
   # Check if all necessary files are available before training
   for DIR_or_file in $DATA_DIR $RESULTS_DIR $BERT_CONFIG ${PHASE1_CKPT}.meta; do
      if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
         echo "Error! $DIR_or_file directory missing. Please mount correctly"
         exit -1
      fi
   done
}

#run_per_ip check_dirs || exit -1

if $eval_only
then
  echo -------------------------------------------------------------------------
  echo "Running evaluation"
  echo
  echo "python3 $SCRIPT_DIR/run_pretraining.py"
  echo "    input_files_dir=$INPUT_FILES"
  echo "    init_checkpoint=$PHASE1_CKPT"
  echo "    eval_files_dir=$EVAL_FILES"
  echo "    output_dir=$RESULTS_DIR_PHASE2"
  echo "    bert_config_file=$BERT_CONFIG"
  echo "    do_train=False"
  echo "    do_eval=True"
  echo "    train_batch_size=$train_batch_size_phase2"
  echo "    eval_batch_size=$eval_batch_size"
  echo "    max_eval_steps=$max_eval_steps"
  echo "    max_seq_length=$seq_len"
  echo "    max_predictions_per_seq=$max_pred_per_seq"
  echo "    num_train_steps=$train_steps_phase2"
  echo "    num_accumulation_steps=$num_accumulation_steps_phase2"
  echo "    num_warmup_steps=$warmup_steps_phase2"
  echo "    save_checkpoints_steps=$save_checkpoints_steps"
  echo "    learning_rate=$learning_rate_phase2"
  echo "    $horovod_str $PREC"
  echo "    allreduce_post_accumulation=True"
  echo "    enable_device_warmup=0"
  echo "    samples_between_eval=$samples_between_eval"
  echo "    stop_threshold=$stop_threshold"
  echo "    samples_start_eval=$samples_start_eval"
  echo "    is_dist_eval_enabled=$is_dist_eval_enabled"
  echo "    dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json"
  echo -------------------------------------------------------------------------
#  time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
  time horovodrun -np 8 python3 ./run_pretraining.py \
      --input_files_dir=$INPUT_FILES \
      --init_checkpoint=$PHASE1_CKPT \
      --eval_files_dir=$EVAL_FILES\
      --output_dir=$RESULTS_DIR_PHASE2 \
      --bert_config_file=$BERT_CONFIG \
      --do_train=False \
      --do_eval=True \
      --train_batch_size=$train_batch_size_phase2 \
      --eval_batch_size=$eval_batch_size \
      --max_eval_steps=$max_eval_steps \
      --max_seq_length=$seq_len \
      --max_predictions_per_seq=$max_pred_per_seq \
      --num_train_steps=$train_steps_phase2 \
      --num_accumulation_steps=$num_accumulation_steps_phase2 \
      --num_warmup_steps=$warmup_steps_phase2 \
      --save_checkpoints_steps=$save_checkpoints_steps \
      --learning_rate=$learning_rate_phase2 \
      $horovod_str $PREC \
      --allreduce_post_accumulation=True \
      --enable_device_warmup=0 \
      --samples_between_eval=$samples_between_eval \
      --stop_threshold=$stop_threshold \
      --samples_start_eval=$samples_start_eval \
      --is_dist_eval_enabled=$is_dist_eval_enabled \
      --dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json
else
  echo -------------------------------------------------------------------------
  echo "Running the Pre-Training :: Phase 2"
  echo
  echo "python3 $SCRIPT_DIR/run_pretraining.py"
  echo "    input_files_dir=$INPUT_FILES"
  echo "    init_checkpoint=$PHASE1_CKPT"
  echo "    eval_files_dir=$EVAL_FILES"
  echo "    output_dir=$RESULTS_DIR_PHASE2"
  echo "    bert_config_file=$BERT_CONFIG"
  echo "    do_train=True"
  echo "    do_eval=False"
  echo "    is_dist_eval_enabled=$is_dist_eval_enabled"
  echo "    train_batch_size=$train_batch_size_phase2"
  echo "    eval_batch_size=$eval_batch_size"
  echo "    max_eval_steps=$max_eval_steps"
  echo "    max_seq_length=$seq_len"
  echo "    max_predictions_per_seq=$max_pred_per_seq"
  echo "    num_train_steps=$train_steps_phase2"
  echo "    num_accumulation_steps=$num_accumulation_steps_phase2"
  echo "    num_warmup_steps=$warmup_steps_phase2"
  echo "    save_checkpoints_steps=$save_checkpoints_steps"
  echo "    learning_rate=$learning_rate_phase2"
  echo "    $horovod_str $PREC"
  echo "    allreduce_post_accumulation=True"
  echo "    enable_device_warmup=True"
  echo "    samples_between_eval=$samples_between_eval"
  echo "    stop_threshold=$stop_threshold"
  echo "    samples_start_eval=$samples_start_eval"
  echo "    dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json"
  echo -------------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=0

python3 ./run_pretraining.py \
      --input_files_dir=$INPUT_FILES \
      --init_checkpoint=$PHASE1_CKPT \
      --eval_files_dir=$EVAL_FILES\
      --output_dir=$RESULTS_DIR_PHASE2 \
      --bert_config_file=$BERT_CONFIG \
      --do_train=True \
      --do_eval=False \
      --is_dist_eval_enabled=$is_dist_eval_enabled \
      --train_batch_size=$train_batch_size_phase2 \
      --eval_batch_size=$eval_batch_size \
      --max_eval_steps=$max_eval_steps \
      --max_seq_length=$seq_len \
      --max_predictions_per_seq=$max_pred_per_seq \
      --num_train_steps=$train_steps_phase2 \
      --num_accumulation_steps=$num_accumulation_steps_phase2 \
      --num_warmup_steps=$warmup_steps_phase2 \
      --save_checkpoints_steps=$save_checkpoints_steps \
      --learning_rate=$learning_rate_phase2 \
      $horovod_str $PREC \
      --allreduce_post_accumulation=True \
      --enable_device_warmup=True \
      --samples_between_eval=$samples_between_eval \
      --stop_threshold=$stop_threshold \
      --samples_start_eval=$samples_start_eval \
      --dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json |& tee -a ${LOGFILE}
fi
