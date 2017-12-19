from chess_zero.config import Config
from chess_zero.agent.model_chess import ChessModel
from multiprocessing import Queue
from threading import Thread
import time
import numpy as np
from collections import namedtuple

QueueItem = namedtuple("QueueItem", "state future")

class ChessModelAPI:
	def __init__(self, config: Config, agent_model: ChessModel):
		self.config = config
		self.agent_model = agent_model
		self.prediction_queue = Queue()
		prediction_worker = Thread(target=self.predict_batch_worker,name="prediction_worker")
		prediction_worker.daemon = True
		prediction_worker.start()

	def predict_batch_worker(self):
		q = self.prediction_queue
		with self.agent_model.graph.as_default():
			while True:
				if q.qsize() > 0:
					item_list = [q.get() for _ in q.qsize()]
					#logger.debug(f"predicting {len(item_list)} items")
					data = np.array([x.state for x in item_list])
					policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
					for item, p, v in zip(item_list, policy_ary, value_ary):
						item.future.set_result((p, float(v)))
				else:
					time.sleep(0.001)

	def please(self, item: QueueItem):
		# assert x.ndim == 4
		# assert x.shape[1:] == (101, 8, 8)
		self.prediction_queue.put(x)