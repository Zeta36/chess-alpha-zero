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

    def canonical_bw_plane(self):
        current_player = self.board.fen().split(" ")[1]
        return black_and_white_plane(self, current_player == 'b')

    def maybe_flip(self, brd, flip = False):
        if flip == False:
            return brd
        # print ("".join( [brd[i : i + 8] for i in reversed(range(0, 64, 8))] ))
        return "".join( [brd[i : i + 8] for i in reversed(range(0, 64, 8))] )

    # this can be used to augment training data (easier) OR dim reduction
    def black_and_white_plane(self, flip = False):
        # flip = True applies the flip + invert color invariant transformation
        one_hot = {}
        one_hot.update(dict.fromkeys(['K', 'k'], [1, 0, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['Q', 'q'], [0, 1, 0, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 1, 0, 0, 0]))
        one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 0, 1, 0, 0]))
        one_hot.update(dict.fromkeys(['N', 'n'], [0, 0, 0, 0, 1, 0]))
        one_hot.update(dict.fromkeys(['P', 'p'], [0, 0, 0, 0, 0, 1]))

        history_p1 = [] #side to move
        history_p2 = [] #side not to move

        # history planes
        for i in range(8):
            if i < len(self.movements):
                board_state = self.replace_tags_board(self.movements[i])
                board_state = self.maybe_flip(board_state.split(" ")[0], flip) 
                history_p1_aux = [one_hot[val] if val.isupper() != flip else [0, 0, 0, 0, 0, 0] \
                    for val in board_state]
                history_p1.append(np.transpose(np.reshape(history_p1_aux, (8, 8, 6)), (2, 0, 1)))
                history_p2_aux = [one_hot[val] if val.islower() != flip else [0, 0, 0, 0, 0, 0] \
                    for val in board_state]
                history_p2.append(np.transpose(np.reshape(history_p2_aux, (8, 8, 6)), (2, 0, 1)))
            else:
                history_p1_aux = [[0, 0, 0, 0, 0, 0] for _ in range(64)]
                history_p1.append(np.transpose(np.reshape(history_p1_aux, (8, 8, 6)), (2, 0, 1)))
                history_p2_aux = [[0, 0, 0, 0, 0, 0] for _ in range(64)]
                history_p2.append(np.transpose(np.reshape(history_p2_aux, (8, 8, 6)), (2, 0, 1)))

        # current state plane
        board_state = self.replace_tags()
        board_state = self.maybe_flip(board_state.split(" ")[0], flip) 
        board_p1 = [one_hot[val] if val.isupper() != flip else [0, 0, 0, 0, 0, 0] \
            for val in board_state]
        history_p1.append(np.transpose(np.reshape(board_p1, (8, 8, 6)), (2, 0, 1)))
        board_p2 = [one_hot[val] if val.islower() != flip else [0, 0, 0, 0, 0, 0] \
            for val in board_state]
        history_p2.append(np.transpose(np.reshape(board_p2, (8, 8, 6)), (2, 0, 1)))

        # one-hot integer plane current player turn, xor if flipped
        current_player = self.board.fen().split(" ")[1]
        current_player = np.full((8, 8), int((current_player == 'w') != flip), dtype=int)

        # fifty move rule number
        fifty_move_number = self.board.fen().split(" ")[4]
        fifty_move_number = np.full((8, 8), int(fifty_move_number), dtype=int)

        return history_p1, history_p2, current_player, fifty_move_number

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
        return self.replace_tags_board(self.board.fen())

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()
