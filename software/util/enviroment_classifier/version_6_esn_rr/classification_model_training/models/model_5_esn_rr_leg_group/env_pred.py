import torch

from models.model_1_esn_rr.env_pred import ESN_RR_Classification


class ESN_RR_LegGroupClassification:
    """
    Front/back leg-group sub-reservoir ESN classifier.

    Wraps two independent ESN_RR_Classification models: one fed only
    front-leg (FR, FL) sensor channels, one fed only back-leg (BR, BL)
    sensor channels. Each sub-model has its own reservoir and ridge-
    regression readout, and predicts terrain independently.

    Reuses the ESN_RR_Classification reservoir/readout implementation
    unchanged (models/model_1_esn_rr/env_pred.py) - this class only
    orchestrates the pair of sub-models.
    """

    def __init__(self, front_kwargs=None, back_kwargs=None):
        front_kwargs = front_kwargs or {}
        back_kwargs = back_kwargs or {}
        self.front_model = ESN_RR_Classification(**front_kwargs)
        self.back_model = ESN_RR_Classification(**back_kwargs)

    def fit(self, X_front, X_back, y):
        self.front_model.fit(X_front, y)
        self.back_model.fit(X_back, y)

    def predict(self, x_front_raw, x_back_raw):
        front_pred, front_conf = self.front_model.predict(x_front_raw)
        back_pred, back_conf = self.back_model.predict(x_back_raw)
        agree = front_pred == back_pred
        return {
            'front_pred': front_pred,
            'front_conf': front_conf,
            'back_pred': back_pred,
            'back_conf': back_conf,
            'agree': agree,
        }

    def save_model(self, filepath="esn_leg_group_model_params.pt"):
        self.front_model.save_model(filepath + ".front")
        self.back_model.save_model(filepath + ".back")

    def load_model(self, filepath="esn_leg_group_model_params.pt"):
        self.front_model.load_model(filepath + ".front")
        self.back_model.load_model(filepath + ".back")

    def get_model_size(self):
        front_size = self.front_model.get_model_size()
        back_size = self.back_model.get_model_size()
        return front_size + back_size
