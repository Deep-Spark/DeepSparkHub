#!/bin/bash

NUM_GPU=1

CONFIG="7b"
BATCH_SIZE=8
MICRO_BATCH_SIZE=1
MAX_LENGTH=4096

TP=1
SP=1
EP=1
PP=1

pp_style="1f1b"

#################### Single-Node #################
nsys profile -o nsys_mixtral_layer -t cuda,cudnn,cublas \
	--capture-range cudaProfilerApi --capture-range-end stop --force-overwrite true \
		torchrun --standalone --nproc_per_node $NUM_GPU benchmark.py \
			-c $CONFIG \
			-b $BATCH_SIZE \
			-l $MAX_LENGTH \
			--mbs $MICRO_BATCH_SIZE \
			--tp $TP \
			--sp $SP \
			--ep $EP \
			--pp $PP \
			--pp_style $pp_style \
			--profile \
			--nsys

# torchrun --standalone --nproc_per_node $NUM_GPU benchmark.py \
# 	-c $CONFIG \
# 	-b $BATCH_SIZE \
# 	-l $MAX_LENGTH \
# 	--mbs $MICRO_BATCH_SIZE \
# 	--tp $TP \
# 	--sp $SP \
# 	--ep $EP \
# 	--pp $PP \
# 	--pp_style $pp_style
	#  > benchmark_mixtral_tp${TP}sp${SP}pp${PP}ep${EP}.log 2>&1
