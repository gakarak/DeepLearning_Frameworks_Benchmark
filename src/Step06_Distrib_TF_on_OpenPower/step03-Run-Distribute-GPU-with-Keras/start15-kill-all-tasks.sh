#!/bin/bash

uid="${USER}"

for ii in `seq 1 9`
do
    echo "(kill) --> ( ${uid}@node${ii} )"
    ssh ${uid}@node${ii} "sudo pkill -9 python"
done
