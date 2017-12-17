import os
from datetime import datetime
from logging import getLogger
from time import time
import chess
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from chess_zero.lib.model_helper import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed
import numpy as np

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.4)
    return SelfPlayWorker(config, env=ChessEnv()).start()


class SelfPlayWorker:
    def __init__(self, config: Config, env=None, model=None):
        """

        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.model = model
        self.env = env     # type: ChessEnv
        self.black = None  # type: ChessPlayer
        self.white = None  # type: ChessPlayer
        self.buffer = []

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        self.buffer = []
        self.idx = 1

        while True:
            start_time = time()
            env = self.start_game(self.idx)
            end_time = time()
            logger.debug(f"game {self.idx} time={end_time - start_time:.3f}s "
                         f"halfmoves={int(env.turn)} {env.winner} "
                         f"{'by resign ' if env.resigned else '          '}"
                         f"{env.observation.split(' ')[0]}")
            if (self.idx % self.config.play_data.nb_game_in_file) == 0:
                reload_best_model_weight_if_changed(self.model)
            self.idx += 1

    def start_game(self, idx):
        self.env.reset()
        self.black = ChessPlayer(self.config, self.model)
        self.white = ChessPlayer(self.config, self.model)
        while not self.env.done:
            if self.env.turn >= self.config.play.max_game_length:
                self.env.adjudicate()
                break
            if self.env.board.turn == chess.BLACK:
                action = self.black.action(self.env)
            else:
                action = self.white.action(self.env)
            #print(action)
            self.env.step(action)
        self.finish_game()
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()
        return self.env

    def save_play_data(self, write=True):

        data = []

        for i in range(len(self.white.moves)):
            data.append(self.white.moves[i])
            if i < len(self.black.moves):
                data.append(self.black.moves[i])

        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model
