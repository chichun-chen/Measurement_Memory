#!/bin/bash


for num in {1..10}
do
    python Experiment_Molecular.py -MM  $num &
done
