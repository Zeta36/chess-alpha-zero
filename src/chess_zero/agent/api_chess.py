from chess_zero.config import Config
from multiprocessing import Manager
from threading import Thread
import time
import numpy as np
from collections import namedtuple

class ChessModelAPI:
	def __init__(self, config: Config, agent_model): # ChessModel
		self.config = config
		self.agent_model = agent_model
		self.prediction_queue = Manager().Queue()
		prediction_worker = Thread(target=self.predict_batch_worker,name="prediction_worker")
		prediction_worker.daemon = True
		prediction_worker.start()

	def predict_batch_worker(self):
		q = self.prediction_queue
		with self.agent_model.graph.as_default():
			while True:
				d, r = q.get()
				data, result_queues = [d], [r]
				for _ in range(q.qsize()):
					d, r = q.get_nowait()
					data.append(d)
					result_queues.append(r)
				#print(f"predicting {len(result_queues)} items")
				data = np.array(data)
				policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
				for r_q, p, v in zip(result_queues, policy_ary, value_ary):
					r_q.put((p, float(v)))