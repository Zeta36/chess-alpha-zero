import enum
import chess.pgn
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


class ChessEnv:
    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False
        self.movements = []

    def reset(self):
        self.board = chess.Board()
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        self.movements = []
        return self

    def update(self, board, history=list()):
        self.board = chess.Board(board)
        self.turn = self.board.fullmove_number
        self.done = False
        self.winner = None
        self.resigned = False
        self.movements = history
        return self

    def set_history(self, history):
        self.movements = history

    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action is None:
            self._resigned()
            return self.board, {}

        self.board.push_uci(action)

        if len(self.movements) > 8:
            self.movements.pop(0)
        self.movements.append(self.board.fen())

        self.turn += 1

        if self.board.is_game_over() or self.board.can_claim_threefold_repetition():
            self._game_over()

        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            result = self.board.result()
            if result == '1-0':
                self.winner = Winner.white
            elif result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def _resigned(self):
        self._win_another_player()
        self._game_over()
        self.resigned = True

    def _win_another_player(self):
        if self.board.turn == chess.BLACK:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def ending_average_game(self):
        self.resigned = True
        self.done = True
        self.winner = Winner.draw

    def black_and_white_plane(self):
        one_hot = {}
        one_hot.update(dict.fromkeys(['K', 'k'], [1, 0, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['Q', 'q'], [0, 1, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 1, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 0, 1, 0, 0]))
        one_hot.update(dict.fromkeys(['N', 'n'], [0, 0, 0, 0, 1, 0]))
        one_hot.update(dict.fromkeys(['P', 'p'], [0, 0, 0, 0, 0, 1]))

        history_white = []
        history_black = []

        # history planes
        for i in range(8):
            if i < len(self.movements):
                board_state = self.replace_tags_board(self.movements[i])
                history_white_aux = [one_hot[val] if val.isupper() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
                history_white.append(np.transpose(np.reshape(history_white_aux, (8, 8, 6)), (2, 0, 1)))
                history_black_aux = [one_hot[val] if val.islower() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
                history_black.append(np.transpose(np.reshape(history_black_aux, (8, 8, 6)), (2, 0, 1)))
            else:
                history_white_aux = [[0, 0, 0, 0, 0, 0] for _ in range(64)]
                history_white.append(np.transpose(np.reshape(history_white_aux, (8, 8, 6)), (2, 0, 1)))
                history_black_aux = [[0, 0, 0, 0, 0, 0] for _ in range(64)]
                history_black.append(np.transpose(np.reshape(history_black_aux, (8, 8, 6)), (2, 0, 1)))

        # current state plane
        board_state = self.replace_tags()
        board_white = [one_hot[val] if val.isupper() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
        history_white.append(np.transpose(np.reshape(board_white, (8, 8, 6)), (2, 0, 1)))
        board_black = [one_hot[val] if val.islower() else [0, 0, 0, 0, 0, 0] for val in board_state.split(" ")[0]]
        history_black.append(np.transpose(np.reshape(board_black, (8, 8, 6)), (2, 0, 1)))

        # one-hot integer plane current player turn
        current_player = self.board.fen().split(" ")[1]
        current_player = np.full((8, 8), int(current_player == 'w'), dtype=int)

        # plane move number
        move_number = self.board.fen().split(" ")[5]
        move_number = np.full((8, 8), int(move_number), dtype=int)

        return history_white, history_black, current_player, move_number

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        env.movements = copy.copy(self.movements)
        return env

    def replace_tags_board(self, board_san):
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")
        return board_san.replace("/", "")

    def replace_tags(self):
        board_san = self.board.fen()
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")
        return board_san.replace("/", "")

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()
