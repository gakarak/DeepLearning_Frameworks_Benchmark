#!/bin/bash

for ii in `seq 1 5`
do
    echo "----->"
    echo "(ps aux) --> ( ubuntu@node${ii}g4 )"
    ssh ubuntu@node${ii}g4 "ps aux | grep python"
    echo "<-----"
done
