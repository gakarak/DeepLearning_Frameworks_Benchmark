#!/bin/bash

numVal=100

fnLabels="labels.txt"
fnTrain="idx-train.txt"
fnVal="idx-val.txt"
fnAll="idx-all.txt"

for ll in `cat $fnLabels`
do
    find ${PWD}/${ll} -name '*.png' | sed "s/$/,$ll/g"
done | shuf > $fnAll

numTot=`cat $fnAll | wc -l`
((numVal=numTot/5))

echo "tot/val = ${numTot}/${numVal}"

cat $fnAll | head -n -${numVal} > $fnTrain
cat $fnAll | tail -n  ${numVal} > $fnVal
