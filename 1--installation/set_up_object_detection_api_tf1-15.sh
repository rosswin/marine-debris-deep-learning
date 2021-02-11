# Script to install the TFODAPI on a Linux VM or a Docker container. 
# It carries out the installation steps described here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# This is a "frozen" script that helps to setup and install the training environment for 
# Ross's Masters Thesis work. This shell script was forked from Microsoft AI for Earth Utils 
# (from https://github.com/microsoft/ai4eutils/tree/master/TF_OD_API) and adapted to 
# run the Ross's preferred Tensorflow version on Ross's hardware.

# You should use the "main" shell script located at the repo above as this file will not 
# ever be updated. Many thanks to the Microsoft AI4E team for their support!

#added by ross jan 20, 2020 to get around interactive install issues with tzdata
export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Pacific/Honolulu /etc/localtime #change the time zone (Pacific/Honolulu) to yours. Google tzdata for more info.
dpkg-reconfigure --frontend noninteractive tzdata

apt-get update -y
apt-get install -y git wget python3-tk

pip install --upgrade pip
pip install tqdm Cython contextlib2 pillow lxml jupyterlab matplotlib geopandas pandas scipy seaborn xmltodict rasterio

cd /lib/tf

#git clone https://github.com/tensorflow/models models  # Dockerfile moves this script to /lib/tf/ so that TFODAPI is installed there

# this is a custom fork of tensorflow/models with modifications made which allow a TFODAPI class confusion matrix to be made.
git clone https://github.com/rosswin/models.git models

#ross commenting this out because we're jumping to tf 1.15
#cd models
#git reset --hard 8367cf6dabe11adf7628541706b660821f397dce  # this is a good commit from 2019/03/06 that works with Python3
#cd ..

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /lib/tf/models/research/
cd ../..

mkdir protoc_3.3
cd protoc_3.3
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
chmod 775 protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip

cd ../models/research

apt-get install -y protobuf-compiler

echo *** Installed protobuf-compiler

../../protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
echo **** PYTHONPATH used to run model_builder_test.py
echo $PYTHONPATH

python setup.py sdist
(cd slim && python setup.py sdist)

echo *********** PWD is
echo $PWD
echo *****
