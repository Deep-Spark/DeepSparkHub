#!/bin/bash

set -e
python3 -m mlperf_logging.package_checker $1 training 0.7.0
python3 -m mlperf_logging.result_summarizer $1 training 0.7.0

