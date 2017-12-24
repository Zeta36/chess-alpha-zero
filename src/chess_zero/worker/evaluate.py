import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper import get_next_generation_model_dirs, pretty_print
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight

logger = getLogger(__name__)

def start(config: Config):
    #tf_util.set_session_config(config.play.vram_frac)
    return EvaluateWorker(config).start()

class EvaluateWorker:
    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

    def start(self):
        while True:
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model)
                self.current_model = ng_model
            self.move_model(model_dir) # i lost my models because of this :(

    def evaluate_model(self, ng_model):
        ng_pipes = self.m.list([ng_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

        futures = []
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=ng_pipes, current_white=(game_idx % 2 == 0))
                futures.append(fut)

            results = []
            for fut in as_completed(futures):
                # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
                ng_score, env, current_white = fut.result()
                results.append(ng_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                logger.debug(f"game {game_idx:3}: ng_score={ng_score:.1f} as {'black' if current_white else 'white'} "
                             f"{'by resign ' if env.resigned else '          '}"
                             f"win_rate={win_rate*100:5.1f}% "
                             f"{env.board.fen().split(' ')[0]}")

                colors = ("current_model", "ng_model")
                if not current_white:
                    colors = reversed(colors)
                pretty_print(env, colors)

                if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                    logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                    return False
                if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug(f"win count reach {results.count(1)} so change best model")
                    return True

        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate*100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        rc = self.config.resource
        # config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        # weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        # os.remove(config_path)
        # os.remove(weight_path)
        new_dir = os.path.join(rc.next_generation_model_dir, "copies", model_dir.name)
        os.rename(model_dir, new_dir)

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

def play_game(config, cur, ng, current_white: bool) -> (float, ChessEnv, bool):
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = ChessEnv().reset()

    current_player = ChessPlayer(config, pipes=cur_pipes, play_config=config.eval.play_config)
    ng_player = ChessPlayer(config, pipes=ng_pipes, play_config=config.eval.play_config)
    if current_white:
        white, black = current_player, ng_player
    else:
        white, black = ng_player, current_player

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        env.step(action)
        if env.num_halfmoves >= config.eval.max_game_length:
            env.adjudicate()

    if env.winner == Winner.draw:
        ng_score = 0.5
    elif env.white_won == current_white:
        ng_score = 0
    else:
        ng_score = 1
    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return ng_score, env, current_white