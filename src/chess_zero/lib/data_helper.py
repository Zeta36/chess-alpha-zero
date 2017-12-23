import os
import ujson
from datetime import datetime
from glob import glob
from logging import getLogger

import chess
import pyperclip
from chess_zero.config import ResourceConfig

logger = getLogger(__name__)


def pretty_print(env, colors):
    new_pgn = open("test3.pgn", "at")
    game = chess.pgn.Game.from_board(env.board)
    game.headers["Result"] = env.result
    game.headers["White"], game.headers["Black"] = colors
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    new_pgn.write(str(game) + "\n\n")
    new_pgn.close()
    pyperclip.copy(env.board.fen())


def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files


def get_game_data_filenames(rc: ResourceConfig):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc: ResourceConfig):
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            ujson.dump(data, f)
    except Exception as e:
        print(e)


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return ujson.load(f)
    except Exception as e:
        print(e)

# def conv_helper(path):
#     with open(path, "rt") as f:
#         data = json.load(f)
#     with open(path, "wb") as f:
#         pickle.dump(data, f)

# def convert_json_to_pickle():
#     import os
#     files = [x for x in os.listdir() if x.endswith(".json")]
#     from concurrent.futures import ProcessPoolExecutor
#     with ProcessPoolExecutor(max_workers=6) as executor:
#         executor.map(conv_helper,files)
