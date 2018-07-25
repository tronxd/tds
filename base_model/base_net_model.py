from base_model.base_model_class import BaseModel
import os

class BaseNetModel(BaseModel):
    def __init__(self, train_params, gpus):
        self.train_params = train_params
        self.gpus = gpus
        self.net_model = None
        self.model_path = None

    def save_weights(self):
        if self.net_model is None:
            raise Exception("Model is not initialized")
        model_path = os.path.join(self.model_path, 'model_weights.hdf5')
        print("Saving model to {}".format(model_path))
        self.net_model.save_weights(model_path)
        print("Model saved!")

    def load_weights(self):
        if self.net_model is None:
            raise Exception("Model is not initialized")

        model_path = os.path.join(self.model_path, 'model_weights.hdf5')
        print("Loading model checkpoint {} \n".format(model_path))
        self.net_model.load_weights(model_path)
        print("Model loaded!")

    def build_model(self):
        raise NotImplementedError()