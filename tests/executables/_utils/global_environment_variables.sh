export PROJECT_DIR=../../
export DRT_MEMCPYUSEKERNEL=20000000000
: ${RUN_MODE:="strict"}
: ${NONSTRICT_EPOCH:=5}

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}
