#!/bin/bash

PYTHON_PATH=$(which python3)
${PYTHON_PATH} -m pip list | grep "^torch .*+corex" || {
  echo "ERROR: building SSD requries the corex torch has been installed."
  exit 1
}

a=$(pip3 show torch|awk '/Version:/ {print $NF}'); b=(${a//+/ }); c=(${b//./ })
if [[ ${c[0]} -eq 1 ]]; then
  rm -rf csrc && ln -s csrc_pt1 csrc
elif [[ ${c[0]} -eq 2 ]]; then
  rm -rf csrc && ln -s csrc_pt2 csrc
else
  echo "ERROR: torch version ${a} is not as expected, please check."
  exit 1
fi


${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit
${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit
rm -rf SSD.egg-info

# Return 0 status if all finished
exit 0
