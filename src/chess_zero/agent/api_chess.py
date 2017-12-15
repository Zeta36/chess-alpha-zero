from chess_zero.config import Config


class ChessModelAPI:
    def __init__(self, config: Config, agent_model):
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (101, 8, 8) or x.shape[1:] == (101, 8, 8)
        is_batch = (x.ndim == 4)
        if is_batch == False:
            x = x.reshape(1, 101, 8, 8)
        policy, value = self.agent_model.model.predict_on_batch(x)

        if is_batch:
            return policy, value
        else: # match input format
            return policy[0], value[0]
