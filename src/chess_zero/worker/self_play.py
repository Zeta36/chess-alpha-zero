import os
from datetime import datetime
from logging import getLogger
from time import time
import chess
from concurrent.futures import ProcessPoolExecutor, as_completed
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.agent.model_chess import ChessModel
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file, prettyprint
from chess_zero.lib.model_helper import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed
from multiprocessing import Manager
from collections import deque

import numpy as np

logger = getLogger(__name__)

def start(config: Config):
    return SelfPlayWorker(config).start()
    # #tf_util.set_session_config(config.play.vram_frac)
    # model = load_model(config)

    # futures = []
    # with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
    #     for _ in range(config.play.max_processes):
    #         alloc_pipes = model.get_pipes(config.play.search_threads)
    #         futures.append(executor.submit(startone, config, alloc_pipes))

    # return [f.result() for f in futures]


# def startone(config, pipes: list):
#     SelfPlayWorker(config, pipes).start()

class SelfPlayWorker:
    def __init__(self, config: Config):
        """
        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)])

    def start(self):
        self.buffer = []

        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for game_idx in range(self.config.play.max_processes):
                futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes))
            game_idx = 0
            while True:
                game_idx += 1
                start_time = time()
                env, data = futures.popleft().result()
                print(f"game {game_idx:3} time={time() - start_time:5.1f}s "
                    f"halfmoves={env.num_halfmoves:3} {env.winner:12} "
                    f"{'by resign ' if env.resigned else '          '}")

                prettyprint(env, ("current_model", "current_model"))
                self.buffer += data
                if (game_idx % self.config.play_data.nb_game_in_file) == 0:
                    self.flush_buffer()
                    reload_best_model_weight_if_changed(self.current_model)
                futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes)) # Keep it going

        if len(data) > 0:
            self.flush_buffer()

    def load_model(self) -> list:
        model = ChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model
    # def start_game(self, idx):
    #     self.env.reset()
    #     self.black = ChessPlayer(self.config, pipes=self.pipes)
    #     self.white = ChessPlayer(self.config, pipes=self.pipes)
    #     while not self.env.done:
    #         if self.env.board.turn == chess.WHITE:
    #             action = self.white.action(self.env)
    #         else:
    #             action = self.black.action(self.env)
    #         #print(action)
    #         self.env.step(action)
    #         if self.env.num_halfmoves >= self.config.play.max_game_length:
    #             self.env.adjudicate()
    #     self.finish_game()
    #     self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
    #     self.remove_play_data()
    # #     return self.env

    # def save_play_data(self):
    #     rc = self.config.resource
    #     game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    #     path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
    #     logger.info(f"save play data to {path}")
    #     write_game_data_to_file(path, self.buffer)
    #     self.buffer = []

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target = write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self):
        return
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    # def finish_game(self):
    #     if self.env.winner == Winner.black:
    #         black_win = 1
    #     elif self.env.winner == Winner.white:
    #         black_win = -1
    #     else:
    #         black_win = 0

    #     self.black.finish_game(black_win)
    #     self.white.finish_game(-black_win)


def self_play_buffer(config, cur) -> (ChessEnv, list):
    pipes = cur.pop() # borrow
    env = ChessEnv().reset()

    white = ChessPlayer(config, pipes=pipes)
    black = ChessPlayer(config, pipes=pipes)

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        env.step(action)
        if env.num_halfmoves >= config.play.max_game_length:
            env.adjudicate()

    if env.winner == Winner.white:
        black_win = -1
    elif env.winner == Winner.black:
        black_win = 1
    else:
        black_win = 0

    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    cur.append(pipes)
    return env, data