#!/usr/bin/env bash

while [ `nvidia-smi --query-compute-apps=pid --format=csv | awk 'NR>1' | wc -L` -gt 0 ]
do
echo "waiting another task to be finished"
sleep 10m
done

cd ~/slim && sh scripts/batch_size.sh "0,1"
