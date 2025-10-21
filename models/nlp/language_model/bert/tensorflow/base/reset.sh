#!/bin/bash

# clean cache for host memory
echo 3 > /proc/sys/vm/drop_caches

# reset BI
ixsmi -r

