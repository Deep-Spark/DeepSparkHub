EXIT_CODE=0
rust_path=`command -v rustup`
if [ -z "${rust_path}" ]; then
    echo "Rust not found, try to install rust."
else
    echo "Found rust, skip install process."
    exit ${EXIT_CODE}
fi

echo "Create tmp dir"
echo "Enter rust_tmp_dir"
if [ ! -d "./packages" ]; then
    mkdir rust_tmp_dir
fi
cd rust_tmp_dir

# Download rust
sys_name_str=`uname -a`
if [[ "${sys_name_str}" =~ "aarch64" ]]; then
    echo "Download rust init file"
    wget https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init
    if [[ $? != 0 ]]; then
      echo "ERROR: Download rust failed"
      EXIT_CODE=1
    fi
else
    echo "Error: Install rust fail."
    EXIT_CODE=1
fi

# Install rust
if [ "${EXIT_CODE}" == "0" ]; then
    export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
    export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
    chmod +x rustup-init
    echo "Installing rust"
    ./rustup-init -y

    source $HOME/.cargo/env
    echo "Install rust success."
fi
cd ..
rm -rf rust_tmp_dir
echo "Exit rust_tmp_dir"