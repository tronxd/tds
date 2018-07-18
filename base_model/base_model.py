import os
class BaseModel(object):
    def __init__(self):
        raise NotImplementedError()

    def preprocess_train(self, iq_data):
        raise NotImplementedError()

    def test_model(self, iq_data):
        # splits iq_data to basic block
        raise NotImplementedError()

    def predict_score(self, iq_data_basic_block):
        # call predict_basic_block and does voting
        raise NotImplementedError()

    def plot_prediction(self, iq_data_basic_block):
        # call prediction_matrix and plots it nicely
        raise NotImplementedError()

    def predict_basic_block(self, iq_data_basic_block):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()
