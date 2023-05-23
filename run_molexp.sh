#!/bin/bash


for num in {1..10}
do
    taskset -c $[num-1] python Experiment_Molecular.py -MM $num &
done

for num in {1..10}
do
    taskset -c $[num+10] python Experiment_Molecular.py $num &
done
