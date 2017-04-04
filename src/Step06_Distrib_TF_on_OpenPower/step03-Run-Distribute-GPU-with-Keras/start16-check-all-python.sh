#!/bin/bash

uid="${USER}"

for ii in `seq 1 9`
do
    echo "----->"
    echo "(ps aux) --> ( ${uid}@node${ii} )"
    ssh ${uid}@node${ii} "ps aux | grep python"
    echo "<-----"
done
