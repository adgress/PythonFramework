#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "No directory supplied"
    exit
fi
s1="/data/home/ubuntu/PythonFramework/$1/*"
s2="/home/ubuntu/PythonFramework/$1/"
#echo "$s1"
#echio "$s2"
#ls $s2
starcluster get mpi --node master $s1 $s2
