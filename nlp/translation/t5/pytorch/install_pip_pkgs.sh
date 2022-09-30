#!/bin/bash
PIPCMD=pip3
: ${PKGS_CACHE_DIR:="__null__"}

function install_pip_pkgs() {
    for pkg in "$@"
    do
        if [ ! -d $PKGS_CACHE_DIR ]; then
            $PIPCMD install $pkg
        else
            $PIPCMD install --no-index --find-links=$PKGS_CACHE_DIR $pkg
        fi
    done
}

# Exeample
# pkgs=(1 2 3)
# install_pip_pkgs "${pkgs[@]}"