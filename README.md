# PythonFramework

Python experiment framework

WINDOWS SETUP

Install Python xy
Update path with python install directory

pip --upgrade pip
pip install mpi4py
pip install boto
pip install dccp

NOTE: I had trouble using the latest version of mpi4py on Linux.  See step (3) below.

NOTE: I tried updated scipy, numpy, cvxpy but got errors about not having blas/lapack.  I worked on this but was unable to get it working

Install MPI: https://www.microsoft.com/en-us/download/confirmation.aspx?id=47259 (Path should be updated automatically)


LINUX SETUP

1) Install Anaconda: Many of the necessary packages are in Anaconda

Follow instructions at: https://www.continuum.io/downloads

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

nosetests cvxpy
git clone https://github.com/adgress/PythonFramework.git

#Note: you may need to run this - https://github.com/ContinuumIO/anaconda-issues/issues/445
conda install libgfortran
#Note: you may also need to do this:
conda install numpy
conda install scipy
#Depending on various fortran/blas/lapack issues you may need to do this:
conda install [VARIOUS BLAS LAPACK STUFF]
pip install --upgrade cvxpy

#May also need to do this - https://github.com/conda/conda/issues/1051
apt-get install libsm6 libxrender1 libfontconfig1

#May need to do this
conda reinstall cvxopt
conda reinstall cvxpy

#You may need to do the following if you get an error message about libgfortran missing
#From: https://github.com/ContinuumIO/anaconda-issues/issues/686
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3

3) Install mpi4py

#NOTE: it seems like just running the following will suffice:
conda install mpi4py

#NOTE: After doing this I had to use 'mpirun' instead of 'mpiexec'

You may need to install openmpi and update LD_LIBRARY_PATH

I had trouble with mpi4py version 2.0.0, so I install the previous version instead (1.4.3?)
This can be done using 'pip install mpi4py=X.Y.Z'



4) Copy (or generate) datasets

The raw data and split data is not included in the repository, so it needs to be generated or copied over