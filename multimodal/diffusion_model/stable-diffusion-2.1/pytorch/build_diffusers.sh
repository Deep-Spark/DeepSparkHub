SCRIPTPATH=$(dirname $(realpath "$0"))
cd $(dirname $(realpath "$0"))
COREX_VERSION=${COREX_VERSION:-latest}
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)

export MAX_JOBS=${MAX_JOBS}

echo "Python cmd1: ${PYTHON_PATH} setup.py build"
${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi

export LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"

echo "Python cmd2: ${PYTHON_PATH} setup.py bdist_wheel -d build_pip"
${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0