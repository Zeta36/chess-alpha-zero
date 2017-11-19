from logging import getLogger

from chess_zero.agent.player_chess import HistoryItem
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.lib.model_helpler import load_best_model_weight
import chess

logger = getLogger(__name__)


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.human_color = None
        self.observers = []
        self.model = self._load_model()
        self.ai = None  # type: ChessPlayer
        self.last_evaluation = None
        self.last_history = None  # type: HistoryItem

    def start_game(self, human_is_black):
        self.human_color = chess.BLACK if human_is_black else chess.WHITE
        self.ai = ChessPlayer(self.config, self.model)

    def _load_model(self):
        from chess_zero.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        if not load_best_model_weight(model):
            raise RuntimeError("best model not found!")
        return model

    def move_by_ai(self, env):
        action = self.ai.action(env.observation)

        self.last_history = self.ai.ask_thought_about(env.observation)
        self.last_evaluation = self.last_history.values[self.last_history.action]
        logger.debug(f"evaluation by ai={self.last_evaluation}")

        return action

    def move_by_human(self, env):
        while True:
            try:
                movement = input('\nEnter your movement in UCI format(a1a2, b2b6,...): ')
                if chess.Move.from_uci(movement) in env.board.legal_moves:
                    return movement
                else:
                    print("That is NOT a valid movement :(.")
            except:
                print("That is NOT a valid movement :(.")