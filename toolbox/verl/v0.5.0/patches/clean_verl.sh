#!/bin/bash
pip3 uninstall verl -y || true

rm -rf build
rm -rf build_pip
rm -rf verl.egg-info
# Return 0 status if all finished
exit 0