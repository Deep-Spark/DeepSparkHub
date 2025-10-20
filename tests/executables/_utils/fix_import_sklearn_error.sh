sys_name_str=`uname -a`
if [[ "${sys_name_str}" =~ "aarch64" ]]; then
    if [ -z "$LD_PRELOAD" ]; then
        ligo=`find /usr -iname "libgomp.so.1"`
        for path in $ligo; do
            if [[ "${path}" =~ "libgomp.so.1" ]]; then
                export LD_PRELOAD="${path}"
                echo "Set LD_PRELOAD="${path}""
                break
            fi
        done

        ligo=`find /usr -iname "libgomp-d22c30c5.so.1.0.0"`
        for path in $ligo; do
            if [[ "${path}" =~ "libgomp-d22c30c5.so.1.0.0" ]]; then
                export LD_PRELOAD="${path}"
                echo "Set LD_PRELOAD="${path}""
                break
            fi
        done

    fi
fi