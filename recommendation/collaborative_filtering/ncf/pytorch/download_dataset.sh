
# 新建目录 data
mkdir data 
cd data

# Creates ml-20.zip
curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip

# Unzip
unzip ml-20m.zip

# delete zip
rm -rf ml-20m.zip 

# 返回上一级目录
cd ..
