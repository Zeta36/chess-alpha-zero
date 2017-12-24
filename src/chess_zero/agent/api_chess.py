from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np

from chess_zero.config import Config


class ChessModelAPI:
    # noinspection PyUnusedLocal
    def __init__(self, config: Config, agent_model):  # ChessModel
        self.agent_model = agent_model
        self.pipes = []

    def start(self):
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self):
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def predict_batch_worker(self):
        while True:
            ready = connection.wait(self.pipes,timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))
