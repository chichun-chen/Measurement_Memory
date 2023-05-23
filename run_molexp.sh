#!/bin/bash


for num in {1..10}
do
    taskset -cp 0-7 python Experiment_Molecular.py -MM  $num &
done
