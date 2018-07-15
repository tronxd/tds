from base_model.ae_model import AeModel
from utilities.preprocessing import get_xhdr_sample_rate, load_raw_data, get_basic_block_len
import os

model = AeModel('model\\ae\\CELL_125000')

normal_path = 'iq_data\\CELL\\normal'
anomal_path = 'iq_data\\CELL\\anomal'

normal_records = os.listdir(normal_path)[1:] # discarding train record
anomal_records = os.listdir(anomal_path)


r = anomal_records[3]

print(r)

data_dir = os.path.join(anomal_path, r)
sample_rate = get_xhdr_sample_rate(data_dir)
data_iq = load_raw_data(data_dir)
basic_len = get_basic_block_len(sample_rate)
data_iq = data_iq[:basic_len,:]


# model.train(data_iq, sample_rate)
model.plot_prediction(data_iq, sample_rate)

