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
from chess_zero.lib.model_helper import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed
import random

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.01)
    return SupervisedLearningWorker(config, env=ChessEnv()).start()


class SupervisedLearningWorker:
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
        idx = 1

        while True:
            start_time = time()
            env = self.read_game(idx)
            end_time = time()
            logger.debug(f"game {idx} time={end_time - start_time} sec, "
                         f"turn={int(env.turn/2)}:{env.observation} - Winner:{env.winner} - by resignation?:{env.resigned}")
            if (idx % self.config.play_data.nb_game_in_file) == 0:
                reload_best_model_weight_if_changed(self.model)
            idx += 1

    def read_game(self, idx):
        self.env.reset()
        self.black = ChessPlayer(self.config, self.model)
        self.white = ChessPlayer(self.config, self.model)
        files = find_pgn_files(self.config.resource.play_data_dir)
        if len(files) > 0:
            random.shuffle(files)
            filename = files[0]
            pgn = open(filename, errors='ignore')
            size = os.path.getsize(filename)
            pos = random.randint(0, size)
            pgn.seek(pos)

            line = pgn.readline()
            offset = 0
            # Parse game headers.
            while line:
                if line.isspace() or line.startswith("%"):
                    line = pgn.readline()
                    continue

                # Read header tags.
                tag_match = TAG_REGEX.match(line)
                if tag_match:
                    offset = pgn.tell()
                    break

                line = pgn.readline()

            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            result = game.headers["Result"]
            actions = []
            while not node.is_end():
                next_node = node.variation(0)
                actions.append(node.board().uci(next_node.move))
                node = next_node
            pgn.close()

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

            if self.env.winner != Winner.draw:
                self.finish_game()
                self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
                self.remove_play_data()
        return self.env

    def save_play_data(self, write=True):
        num = min(len(self.white.moves), len(self.black.moves))
        data = [None] * (num * 2)
        data[::2] = self.white.moves[:num]
        data[1::2] = self.black.moves[:num]
        data.extend(self.white.moves[num:])
        data.extend(self.black.moves[num:])
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
