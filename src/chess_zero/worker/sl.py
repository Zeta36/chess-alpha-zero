import os
from datetime import datetime
from logging import getLogger
from time import time
import chess.pgn
import re
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file, find_pgn_files
from threading import Thread

import random

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    #tf_util.set_session_config(per_process_gpu_memory_fraction=0.01)
    return SupervisedLearningWorker(config, env=ChessEnv()).start()


class SupervisedLearningWorker:
    def __init__(self, config: Config, env=None):
        """

        :param config:
        :param ChessEnv|None env:
        :param chess_zero.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.env = env     # type: ChessEnv
        self.black = None  # type: ChessPlayer
        self.white = None  # type: ChessPlayer
        self.buffer = []

    def start(self):
        self.buffer = []
        self.idx = 1
        start_time = time()

        for env in self.read_all_files():
            end_time = time()
            logger.debug(f"game {self.idx:4} time={(end_time - start_time):.3f}s "
                         f"turn={int(env.turn/2)} {env.winner}"
                         f"{' by resign ' if env.resigned else '           '}"
                         f"{env.observation.split(' ')[0]:}")
            start_time=end_time
            self.idx += 1

        self.flush_buffer()

    def read_all_files(self):
        files = find_pgn_files(self.config.resource.play_data_dir)
        print (files)
        from itertools import chain
        return chain.from_iterable(self.read_file(filename) for filename in files)


    def read_file(self,filename):
        pgn = open(filename, errors='ignore')
        for offset, header in chess.pgn.scan_headers(pgn):
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            yield self.add_to_buffer(game)


    def add_to_buffer(self,game):
        self.env.reset()
        self.black = ChessPlayer(self.config)
        self.white = ChessPlayer(self.config)
        result = game.headers["Result"]
        actions = []
        while not game.is_end():
            game = game.variation(0)
            actions.append(game.move.uci())
        k = 0
        observation = self.env.observation
        while not self.env.done and k < len(actions):
            if self.env.board.turn == chess.BLACK:
                action = self.black.sl_action(observation, actions[k])
            else:
                action = self.white.sl_action(observation, actions[k])
            board, info = self.env.step(action)
            observation = board.fen()
            k += 1

        self.env.done = True
        if not self.env.board.is_game_over() and result != '1/2-1/2':
            self.env.resigned = True
        if result == '1-0':
            self.env.winner = Winner.white
        elif result == '0-1':
            self.env.winner = Winner.black
        else:
            self.env.winner = Winner.draw

        self.finish_game()
        self.save_play_data()
        return self.env

    # def read_game(self, idx):
        # self.env.reset()
        # self.black = ChessPlayer(self.config)
        # self.white = ChessPlayer(self.config)
        # files = find_pgn_files(self.config.resource.play_data_dir)
        # if len(files) > 0:
        #     random.shuffle(files)
        #     filename = files[0]
        #     pgn = open(filename, errors='ignore')
        #     size = os.path.getsize(filename)
        #     pos = random.randint(0, size)
        #     pgn.seek(pos)

        #     line = pgn.readline()
        #     offset = 0
        #     # Parse game headers.
        #     while line:
        #         if line.isspace() or line.startswith("%"):
        #             line = pgn.readline()
        #             continue

        #         # Read header tags.
        #         tag_match = TAG_REGEX.match(line)
        #         if tag_match:
        #             offset = pgn.tell()
        #             break

        #         line = pgn.readline()

        #     pgn.seek(offset)
        #     game = chess.pgn.read_game(pgn)
        #     self.add_to_buffer(game)
        #     pgn.close()
        #     self.save_play_data()
        #     self.remove_play_data()
        # return self.env

    def save_play_data(self):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if self.idx % self.config.play_data.sl_nb_game_in_file == 0:
            self.flush_buffer()

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        #print(self.buffer)
        thread = Thread(target = write_game_data_to_file, args=(path,(self.buffer)))
        thread.start()
        self.buffer = []

    # def remove_play_data(self):
    #     files = get_game_data_filenames(self.config.resource)
    #     if len(files) < self.config.play_data.max_file_num:
    #         return
    #     for i in range(len(files) - self.config.play_data.max_file_num):
    #         os.remove(files[i])

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)