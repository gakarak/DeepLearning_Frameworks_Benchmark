#!/bin/bash

for ii in `seq 2 5`
do
    scp -r ./*.sh ubuntu@node${ii}g1:${PWD}/
done
