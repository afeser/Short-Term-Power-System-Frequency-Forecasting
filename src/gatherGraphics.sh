#!/bin/bash


# Gather graphics and store under folders
# Run in root dir of the project

mkdir -p output/gathered
mkdir -p output/gathered/MSE/
mkdir -p output/gathered/MAE/
mkdir -p output/gathered/MAPE/

tests=( test21 test28 test29 test30 test31 )
for prefix in "${tests[@]}"
do
  cd output/LoadTest/${prefix}CSV*
  cp MAEPlot.png ../../../output/gathered/MAE/${prefix}.png
  cp MSEPlot.png ../../../output/gathered/MSE/${prefix}.png
  cp MAPEPlot.png ../../../output/gathered/MAPE/${prefix}.png
  cp Complete.csv ../../../output/gathered/${prefix}.csv
  cp Complete.xlsx ../../../output/gathered/${prefix}.xlsx
  cd ../../../

done

rm output/gathered.tar
tar -cf output/gathered.tar output/gathered
