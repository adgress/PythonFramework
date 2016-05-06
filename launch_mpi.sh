#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "Need to specify number of prcoesses"
    exit
fi
mpiexec -n $1 python run_main_parallel.py
