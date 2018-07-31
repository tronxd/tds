import os
class BaseModel(object):
    def __init__(self):
        raise NotImplementedError()

    # Preprocess raw data and persist scalers
    # Returns the preprocessed data
    def preprocess_train_data(self, iq_data,sample_rate):
        raise NotImplementedError()


    # Preprocess raw data from loaded scalers
    # Returns the preprocessed data
    def preprocess_test_data(self, iq_data,sample_rate):
        raise NotImplementedError()

    # Train model and persist parameters
    def train_data(self,preprocessed_data):
        raise NotImplementedError()

    # Predict with loaded model on entire dataset
    def predict_data(self,iq_data,sample_rate):
        raise NotImplementedError()

    # Return an anomaly score with loaded model on entire dataset
    def predict_score(self,iq_data,sample_rate):
        raise NotImplementedError()

    # Predict with loaded model on basic data block
    def predict_basic_block(self, iq_data_basic_block,sample_rate):
        raise NotImplementedError()

    # Return an anomaly score on basic data block
    def predict_basic_block_score(self, iq_data_basic_block,sample_rate):
        # call predict_basic_block and does voting
        raise NotImplementedError()

    def get_score_methods(self):
        return {'normal': self.predict_basic_block_score}

    # call predict_basic_block and plots it nicely
    def plot_prediction(self, iq_data_basic_block,sample_rate):

        raise NotImplementedError()

    # Persist model parameters
    def save_model(self):
        raise NotImplementedError()

    # Load model parameters
    def load_model(self):
        raise NotImplementedError()
