# Install packages
. install_pip_pkgs.sh

pkgs=('transformers==4.12.5' 'sacrebleu==2.0.0' 'datasets==1.16.1')
install_pip_pkgs "${pkgs[@]}"


# Check transformers
if python3 -c "import transformers" >/dev/null 2>&1
then
    echo "Transformers already installed"
else
    echo "Again try to install transformers package"
    source install_rust.sh
    if [ ! -d "./packages" ]; then
        $PIPCMD install --ignore-installed PyYAML
        $PIPCMD install transformers==4.12.5
    else
        $PIPCMD install --ignore-installed --no-index --find-links=./packages PyYAML
        $PIPCMD install --no-index --find-links=./packages transformers
    fi
fi