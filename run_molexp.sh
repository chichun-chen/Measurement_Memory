#!/bin/bash


for num in {1..10}
do
    taskset -c 60-69 python Experiment_Molecular.py -MM $num &
done

for num in {1..10}
do
    taskset -c 80-89  python Experiment_Molecular.py $num &
done

#taskset -c 1  python Experiment_Molecular.py 1 &
#taskset -c 2  python Experiment_Molecular.py -MM 1 &
