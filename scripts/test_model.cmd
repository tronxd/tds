CALL activate tf-gpu
python lstm_anomaly_detector.py -m test -n iq_data/center98-bw20/0.01sec/normal -a iq_data/center98-bw20/0.01sec/anomalies_sinc_flicker/center96mhz-bw2mhz-power30db --weights-load-path model/lstm_model.hdf5