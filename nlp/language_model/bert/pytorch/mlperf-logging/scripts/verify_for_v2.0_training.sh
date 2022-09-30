#!/bin/bash

set -e

# rcp_bypass and rcp_bert_train_samples packahe checker params
# need to be retrieved at package_checker_params file at top-level submission dir.
PACKAGE_CHECKER_PARAMS=""
PACKAGE_CHECKER_PARAMS_FILE="$1/package_checker_params"
if test -f "$PACKAGE_CHECKER_PARAMS_FILE"; then
  while IFS= read -r line
  do
    PACKAGE_CHECKER_PARAMS="$PACKAGE_CHECKER_PARAMS --$line"
  done < "$PACKAGE_CHECKER_PARAMS_FILE"
fi

python3 -m mlperf_logging.package_checker $1 training 2.0.0 $PACKAGE_CHECKER_PARAMS
python3 -m mlperf_logging.result_summarizer $1 training 2.0.0
python3 -m mlperf_logging.repo_checker $1 training 2.0.0
