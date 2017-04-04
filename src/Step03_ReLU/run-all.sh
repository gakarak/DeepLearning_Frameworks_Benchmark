#!/bin/bash

ls -1d */ | while read ll
do
    echo "-----> [$ll] >"
    pushd $ll
    ./run01-multiplerun-dl-tests-v1.sh
    ./run01-multiplerun-dl-tests-v2.sh
    popd
done

