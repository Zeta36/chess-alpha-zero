from chess_zero.config import Config
from chess_zero.agent.model_chess import ChessModel


class ChessModelAPI:
    def __init__(self, config: Config, agent_model: ChessModel):
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        assert x.ndim == 4
        assert x.shape[1:] == (101, 8, 8)
        with self.agent_model.graph.as_default():
        	return self.agent_model.model.predict_on_batch(x)