import os
from datetime import datetime
from logging import getLogger
from time import sleep
import random

from profilehooks import profile

import numpy as np

from chess_zero.agent.model_chess import ChessModel
from chess_zero.config import Config
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from chess_zero.lib.model_helper import load_best_model_weight
from chess_zero.env.chess_env import ChessEnv, canon_input_planes, check_current_planes, isblackturn
import chess
from concurrent.futures import ProcessPoolExecutor
from collections import deque

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(config.trainer.vram_frac)
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ChessModel
        self.loaded_filenames = set()
        self.loaded_data = deque() # this should just be a ring buffer i.e. queue of length 500,000 in AZ
        self.dataset = None
        self.optimizer = None
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        last_load_data_step = last_save_step = total_steps = self.config.trainer.start_total_steps
        self.load_play_data()

        while True:
            if self.dataset_size < self.config.trainer.min_data_size_to_learn:
                logger.info(f"dataset_size={self.dataset_size} is less than {self.config.trainer.min_data_size_to_learn}")
                sleep(60)
                self.load_play_data()
                continue
            #self.update_learning_rate(total_steps)
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            #if last_save_step + self.config.trainer.save_model_steps < total_steps:
            self.save_current_model()
            last_save_step = total_steps

            # if last_load_data_step + self.config.trainer.load_data_steps < total_steps:
            #     self.load_play_data()
            #     last_load_data_step = total_steps

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        from keras.optimizers import SGD, Adam
        self.optimizer = Adam() #SGD(lr=2e-1, momentum=0.9) # Adam better?
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        self.model.model.compile(optimizer=self.optimizer, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        # for filename in (self.loaded_filenames - set(filenames)):
        #     self.unload_data_of_file(filename)
        #     updated = True

        if updated:
            logger.debug("updating training dataset")
            self.dataset = self.collect_all_loaded_data()

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = [], [], []
        while self.loaded_data:
            s, p, v = self.loaded_data.popleft().result()
            #assert s[0].shape== (18,8,8)
            state_ary.extend(s)
            policy_ary.extend(p)
            value_ary.extend(v)

        state_ary = np.asarray(state_ary,dtype=np.float32)
        policy_ary = np.asarray(policy_ary,dtype=np.float32)
        value_ary = np.asarray(value_ary,dtype=np.float32)
        return state_ary, policy_ary, value_ary


    def load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

    def load_data_from_file(self, filename):
        # try:
        logger.debug(f"loading data from {filename}")
        data = read_game_data_from_file(filename)
        self.loaded_data.append( self.executor.submit(convert_to_cheating_data, data) )### HEEEERE, use with SL
        self.loaded_filenames.add(filename)
        # except Exception as e:
        #     logger.warning(str(e))

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])
    # def unload_data_of_file(self, filename):
    #     logger.debug(f"removing data about {filename} from training set")
    #     self.loaded_filenames.remove(filename)
    #     if filename in self.loaded_data:
    #         del self.loaded_data[filename]

def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    env = ChessEnv().reset()
    for state_fen, policy, value in data:
        move_number = int(state_fen.split(' ')[5])

        state_planes = canon_input_planes(state_fen)
        assert check_current_planes(state_fen, state_planes)

        if isblackturn(state_fen):
            #assert np.sum(policy) == 0
            policy = Config.flip_policy(policy)
        else:
            #assert abs(np.sum(policy) - 1) < 1e-8
            pass

        assert len(policy) == 1968
        assert state_planes.dtype == np.float32
        assert state_planes.shape == (18,8,8) #print(state_planes.shape)
        
        value_certainty = min(25, move_number)/25 # reduces the noise of the opening... plz train faster
        SL_value = value*value_certainty + env.testeval()*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(SL_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)