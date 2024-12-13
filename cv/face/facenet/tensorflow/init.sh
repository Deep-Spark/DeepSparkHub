PY_VERSION=$(python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}')
if [ "$PY_VERSION" == "10" ] || [ "$PY_VERSION" == "8" ] || [ "$PY_VERSION" == "9" ];
then
   pip3 install -r requirements.txt
   pip3 install scipy==1.7.2 
   pip3 install numpy==1.23.5
else
   pip3 install -r requirements.txt
   pip3 install scipy
fi

cd data
wget -q http://files.deepspark.org.cn:880/deepspark/data/datasets/lfw_data.tar.gz
wget -q http://files.deepspark.org.cn:880/deepspark/data/datasets/webface_182_44.tar
tar -zxf lfw_data.tar.gz
tar -xf webface_182_44.tar
cd -
