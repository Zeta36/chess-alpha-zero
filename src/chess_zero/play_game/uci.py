import sys
from logging import getLogger

import chess
from chess_zero.config import Config, PlayWithHumanConfig
from chess_zero.play_game.game_model import PlayWithHuman
from chess_zero.env.chess_env import ChessEnv
from random import random

logger = getLogger(__name__)

_DEBUG_ = False

def start(config: Config):
    if _DEBUG_:
        ff = open('helloworld.txt','w')

    PlayWithHumanConfig().update_play_config(config.play)

    chess_model = None
    env = ChessEnv().reset()

    while True:
        line=input()
        if _DEBUG_:
            ff.write(line+'\n')
            ff.flush()
        words=line.rstrip().split(" ",1)
        if words[0] == "uci":
            print("id name ChessZero")
            print("id author ChessZero")
            print("uciok")
        elif words[0]=="isready":
            if(chess_model == None):
                chess_model = PlayWithHuman(config)
            print("readyok")
        elif words[0]=="ucinewgame":
            env.reset()
        elif words[0]=="position":
            words=words[1].split(" ",1)
            #print(words)
            if words[0]=="startpos":
                env.update(chess.STARTING_FEN)
            else:
                env.update(words[0])
            if(len(words)>1):
                words=words[1].split(" ",1)
                if words[0]=="moves":
                    for w in words[1].split(" "):
                        env.step(w)
                        #env.render()
        elif words[0]=="go":
            action = chess_model.move_by_ai(env)
            if _DEBUG_:
                ff.write(f">bestmove {action}\n")
                ff.flush()
            print(f"bestmove {action}")
        elif words[0]=="stop":
            pass #lol
        elif words[0]=="quit":
            break
