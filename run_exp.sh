#!/bin/bash


for num in {1..5}
do
    #taskset -c 0-15 python Experiment_Molecular.py -MM $num &
    taskset -c 0-15 python Experiment_Ising.py -MM $num &
done

for num in {1..5}
do
    #taskset -c 16-31  python Experiment_Molecular.py $num &
    taskset -c 16-31 python Experiment_Ising.py $num &
done

#taskset -c 0  python Experiment_Ising.py 1 &
#taskset -c 1  python Experiment_Ising.py 2 &
#taskset -c 2  python Experiment_Ising.py 3 &

#taskset -c 3  python Experiment_Ising.py -MM 1 &
#taskset -c 4  python Experiment_Ising.py -MM 2 &
#taskset -c 5  python Experiment_Ising.py -MM 3 &
