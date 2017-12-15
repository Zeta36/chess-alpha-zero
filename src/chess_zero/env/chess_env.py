import enum
import chess.pgn
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


class ChessEnv:

    one_hot = {}
    one_hot.update(dict.fromkeys(['K', 'k'], [1, 0, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['Q', 'q'], [0, 1, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 1, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 0, 1, 0, 0]))
    one_hot.update(dict.fromkeys(['N', 'n'], [0, 0, 0, 0, 1, 0]))
    one_hot.update(dict.fromkeys(['P', 'p'], [0, 0, 0, 0, 0, 1]))

    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = chess.Board()
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = chess.Board(board)
        self.turn = self.board.fullmove_number
        self.done = False
        self.winner = None
        self.resigned = False
        return self


    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action is None:
            self._resigned()
            return self.board, {}

        self.board.push_uci(action)

        self.turn += 1

        if self.board.is_game_over(claim_draw=True):
            self._game_over()

        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            result = self.board.result(claim_draw = True)
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

    def all_input_planes(self, flip=False):
        myboard = self.maybe_flip_fen(self.board.fen(), flip)
        current_aux_planes = self.aux_planes(myboard)

        history_both = self.black_and_white_plane(flip)

        ret = np.vstack((history_both, current_aux_planes))
        assert ret.shape == (101, 8, 8)
        return ret

    def canonical_input_planes(self):
        current_player = self.board.fen().split(" ")[1]
        flip = (current_player == 'b')
        return self.all_input_planes(flip)

    @staticmethod
    def maybe_flip_fen(fen, flip = False):
        if flip == False:
            return fen
        foo = fen.split(' ')
        foop = ChessEnv.replace_tags_board(foo[0])
        def swapcase(a):
            return a.lower() if a.isupper() else a.upper()
        return "".join( [foop[i : i + 8] for i in reversed(range(0, 64, 8))] ) \
            + " " + ('w' if foo[1]=='b' else 'b') \
            + " " + "".join( ['-' if a == '-' else swapcase(a) for a in foo[2]] ) \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    @staticmethod
    def to_planes(fen):
        board_state = ChessEnv.replace_tags_board(fen)
        pieces_p1 = [ChessEnv.one_hot[val] if val.isupper() else [0, 0, 0, 0, 0, 0] \
            for val in board_state]
        pieces_p1 = np.transpose(np.reshape(pieces_p1, (8, 8, 6)), (2, 0, 1))
        pieces_p2 = [ChessEnv.one_hot[val] if val.islower() else [0, 0, 0, 0, 0, 0] \
            for val in board_state]
        pieces_p2 = np.transpose(np.reshape(pieces_p2, (8, 8, 6)), (2, 0, 1))
        assert pieces_p1.shape == (6, 8, 8)
        state = np.vstack((pieces_p1, pieces_p2))
        assert state.shape == (12, 8, 8)
        return state

    @staticmethod
    def aux_planes(fen):
        foo = fen.split(' ')
        castling_planes = [ np.full((8,8), int('K' in foo[2])) ]
        castling_planes.append( np.full((8,8), int('Q' in foo[2])))
        castling_planes.append( np.full((8,8), int('k' in foo[2])))
        castling_planes.append( np.full((8,8), int('q' in foo[2])))
        castling_planes = np.asarray(castling_planes)
        assert castling_planes.shape == (4,8,8)
        fifty_move_number = foo[4]
        fifty_move_plane = [np.full((8, 8), int(fifty_move_number), dtype=int)]
        ret = np.vstack((castling_planes, fifty_move_plane))
        assert ret.shape == (5,8,8)
        return ret

    def black_and_white_plane(self, flip = False):
        # flip = True applies the flip + invert color invariant transformation

        history_both = []
        history_moves = []

        # history planes
        for i in range(8):
            board_fen = self.maybe_flip_fen(self.board.fen(),flip)
            history_both.extend(self.to_planes(fen = board_fen))
            if len(self.board.move_stack) > 0:
                history_moves.append(self.board.pop())

        for mov in reversed(history_moves):
            self.board.push(mov)
        history_both = np.asarray(history_both)
        assert history_both.shape == (96, 8, 8)
        return history_both

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    @staticmethod
    def replace_tags_board(board_san):
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
        return ChessEnv.replace_tags_board(self.board.fen())

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    def deltamove(self, fen_next):
        moves = [x for x in self.board.legal_moves]
        for mov in moves:
            self.board.push(mov)
            fee = self.board.fen()
            self.board.pop()
            if fee == fen_next:
                return mov.uci()
        return None
