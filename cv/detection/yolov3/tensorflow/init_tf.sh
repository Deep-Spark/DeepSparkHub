
# Install packages
. install_pip_pkgs.sh

pkgs=('Pillow' 'wget' 'seaborn' 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm' )
install_pip_pkgs "${pkgs[@]}"

