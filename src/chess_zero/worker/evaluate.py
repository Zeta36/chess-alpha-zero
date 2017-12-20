import os
from logging import getLogger
from time import sleep
import chess
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib import tf_util
from chess_zero.lib.data_helper import get_next_generation_model_dirs
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight
import pyperclip

logger = getLogger(__name__)


def start(config: Config):
    tf_util.set_session_config(config.eval.vram_frac)
    return EvaluateWorker(config).start()


class EvaluateWorker:
    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.current_model = None

    def start(self):
        self.current_model = self.load_current_model()

        while True:
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model)
                self.current_model = ng_model
            self.remove_model(model_dir)

    def evaluate_model(self, ng_model):
        new_pgn = open("test.pgn","wt")
        results = []
        win_rate = 0
        for game_idx in range(self.config.eval.game_num):
            # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
            current_white = (game_idx % 2 == 0)
            ng_score, env = self.play_game(self.current_model, ng_model, current_white)
            results.append(ng_score)
            win_rate = sum(results) / len(results)
            logger.debug(f"game {game_idx:3}: ng_score={ng_score:.1f} as {'black' if current_white else 'white'} "
                         f"{'by resign ' if env.resigned else '          '}"
                         f"win_rate={win_rate*100:5.1f}% "
                         f"{env.board.fen()}")
            game = chess.pgn.Game.from_board(env.board)
            game.headers["White"] = "current_model" if current_white else "ng_model"
            game.headers["Black"] = "ng_model" if current_white else "current_model"
            new_pgn.write(str(game)+"\n\n")
            new_pgn.flush()
            pyperclip.copy(env.board.fen())

            if results.count(0) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                break
            if results.count(1) >= self.config.eval.game_num * self.config.eval.replace_rate:
                logger.debug(f"win count reach {results.count(1)} so change best model")
                break

        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate*100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def play_game(self, current_model, ng_model, current_white: bool) -> (float, ChessEnv):
        env = ChessEnv().reset()

        current_player = ChessPlayer(self.config, model=current_model, play_config=self.config.eval.play_config)
        ng_player = ChessPlayer(self.config, model=ng_model, play_config=self.config.eval.play_config)
        if current_white:
            white, black = current_player, ng_player
        else:
            white, black = ng_player, current_player

        while not env.done:
            if env.turn >= self.config.eval.max_game_length:
                env.adjudicate()
                break
            if env.board.turn == chess.BLACK:
                action = black.action(env)
            else:
                action = white.action(env)
            env.step(action)

        if env.winner == Winner.draw:
            ng_score = 0.5
        elif (env.winner == Winner.white) == current_white:
            ng_score = 0
        else:
            ng_score = 1
        return ng_score, env

    def load_current_model(self):
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info("There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)
