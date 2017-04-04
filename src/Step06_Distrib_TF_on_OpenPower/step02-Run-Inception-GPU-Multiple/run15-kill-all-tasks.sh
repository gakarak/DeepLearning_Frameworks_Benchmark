#!/bin/bash

for ii in `seq 1 5`
do
    echo "(kill) --> ( ubuntu@node${ii}g4 )"
    ssh ubuntu@node${ii}g4 "sudo pkill -9 python"
done
