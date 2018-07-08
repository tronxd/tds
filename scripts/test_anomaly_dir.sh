#!/usr/bin/env bash
dirs=($(find "$1" -mindepth 1 -type d  ))
for dir  in "${dirs[@]}" ; do
	python lstm_anomaly_detector.py -m test -a "$dir" --weights-load-path lstm_gps_weights.hdf5
done 
