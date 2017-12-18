from chess_zero.config import Config


class ChessModelAPI:
    def __init__(self, config: Config, agent_model):
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        assert x.ndim == 4
        assert x.shape[1:] == (101, 8, 8)
        return self.agent_model.model.predict_on_batch(x)