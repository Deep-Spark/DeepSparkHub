#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python3 setup.py install 

# EMD
cd $HOME/extensions/emd
python3 setup.py install 
