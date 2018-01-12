"""
Utility methods for playing an actual game as a human against a model.
"""

import sys
from logging import getLogger

from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config, PlayWithHumanConfig
from chess_zero.env.chess_env import ChessEnv

logger = getLogger(__name__)

# noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
def start(config: Config):

    PlayWithHumanConfig().update_play_config(config.play)

    me_player = None
    env = ChessEnv().reset()

    while True:
        line = input()
        words = line.rstrip().split(" ",1)
        if words[0] == "uci":
            print("id name ChessZero")
            print("id author ChessZero")
            print("uciok")
        elif words[0] == "isready":
            if not me_player:
                me_player = get_player(config)
            print("readyok")
        elif words[0] == "ucinewgame":
            env.reset()
        elif words[0] == "position":
            words = words[1].split(" ",1)
            if words[0] == "startpos":
                env.reset()
            else:
                if words[0] == "fen": # skip extraneous word
                    words = words[1].split(' ',1)
                fen = words[0]
                for _ in range(5):
                    words = words[1].split(' ',1)
                    fen += " " + words[0]
                env.update(fen)
            if len(words) > 1:
                words = words[1].split(" ",1)
                if words[0] == "moves":
                    for w in words[1].split(" "):
                        env.step(w, False)
        elif words[0] == "go":
            if not me_player:
                me_player = get_player(config)
            action = me_player.action(env, False)
            print(f"bestmove {action}")
        elif words[0] == "stop":
            pass
        elif words[0] == "quit":
            break


def get_player(config):
    from chess_zero.agent.model_chess import ChessModel
    from chess_zero.lib.model_helper import load_best_model_weight
    model = ChessModel(config)
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")
    return ChessPlayer(config, model.get_pipes(config.play.search_threads))


def info(depth, move, score):
    print(f"info score cp {int(score*100)} depth {depth} pv {move}")
    sys.stdout.flush()
