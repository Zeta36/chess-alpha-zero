import os
from glob import glob
import fnmatch
from logging import getLogger
import json

from chess_zero.config import ResourceConfig

logger = getLogger(__name__)


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
            json.dump(data, f)
    except Exception as e:
        print(e)

def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
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