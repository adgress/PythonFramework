# PythonFramework

Python experiment framework

LINUX SETUP

1) Install Anaconda: Many of the necessary packages are in Anaconda

Follow instructions at: https://www.continuum.io/downloads

LINUX:

wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh
bash Anaconda3-2.4.1-Linux-x86_64.sh  -b

NOTE: There seems to be an issue with the installation of python on overachiever (a function seems to be missing from subprocesss)
Hence, installing Anaconda seems to be necessary

NOTE: You may need to create a .profile file with the following to load .bashrc on login:

if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi


2) Install cvxpy

Follow instructions at: http://www.cvxpy.org/en/latest/install/

NOTE: With Anaconda you probably just need to run:

sudo apt-get update
sudo apt-get -y install git libatlas-base-dev gfortran python-dev python-pip python-numpy python-scipy python-nose

#Note that this DOES NOT use sudo
pip install cvxpy

#Note: I think this can be installed before cvxpy, not sure though
#sudo apt-get -y install python-nose

nosetests cvxpy
git clone https://github.com/adgress/PythonFramework.git


3) Copy (or generate) datasets

The raw data and split data is not included in the repository, so it needs to be generated or copied over