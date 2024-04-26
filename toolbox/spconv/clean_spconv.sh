#!/bin/bash

rm -rf build
rm -rf build_pip
rm -rf spconv.egg-info
pip3 uninstall spconv -y

# Return 0 status if all finished
exit 0
