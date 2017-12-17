import enum
import chess.pgn
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


class ChessEnv:
    plane_order = ['K','Q','R','B','N','P','k','q','r','b','n','p']
    ind = {}

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
        self.done = False
        self.winner = None
        self.resigned = False
        return self


    def step(self, action: str, check_over = True):
        """
        :param int|None action, None is resign
        :return:
        """
        if check_over and action is None:
            self._resigned()
            return self.board, {}

        self.board.push_uci(action)

        self.turn += 1

        if check_over and self.board.is_game_over(claim_draw=True):
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

    def adjudicate(self):
        self.resigned = False
        self.done = True
        if abs(self.testeval()) < 0.01:
            self.winner= Winner.draw
        elif self.testeval() > 0:
            self.winner = (Winner.white if self.board.turn == chess.WHITE else Winner.black)
        else:
            self.winner = (Winner.black if self.board.turn == chess.WHITE else Winner.white)

    def ending_average_game(self):
        self.resigned = False
        self.done = True
        self.winner = Winner.draw

    def testeval(self) -> float:
        piecevals = {'K': 64, 'Q': 9, 'R':5,'B':3.25,'N':3,'P':1}
        ans = 0.0
        tot = 0
        for c in self.board.fen().split(' ')[0]:
            if not c.isalpha():
                continue
            assert c.upper() in piecevals   
            if c.isupper():
                ans += piecevals[c]
                tot += piecevals[c]
            else:
                ans -= piecevals[c.upper()]
                tot += piecevals[c.upper()]
        v = ans/tot
        if self.board.turn == chess.BLACK:
            v = -v
        assert abs(v) <= 1
        return v

    def canonical_input_planes(self):
        current_player = self.board.fen().split(" ")[1]
        flip = (current_player == 'b')
        return self.all_input_planes(flip)

    def all_input_planes(self, flip=False):
        myboard = self.maybe_flip_fen(self.board.fen(), flip)
        current_aux_planes = self.aux_planes(myboard)

        history_both = self.black_and_white_plane(flip)

        ret = np.vstack((history_both, current_aux_planes))
        assert ret.shape == (101, 8, 8)
        return ret

    def check_current_planes(self, planes):
        parts = np.split(planes,[96,5])
        hist_both = np.asarray(np.split(parts[0],8)) # 8 histories
        cur = hist_both[0]
        assert cur.shape == (12, 8, 8)
        fakefen = ["1" for _ in range(64)]
        for i in range(12):
            for rank in range(8):
                for file in range(8):
                    if cur[i][rank][file] == 1:
                        assert fakefen[rank * 8 + file] == '1'
                        fakefen[rank * 8 + file] = ChessEnv.plane_order[i]

        realfen = self.board.fen()
        if self.board.turn == chess.BLACK:
            realfen = ChessEnv.maybe_flip_fen(realfen, flip=True)
        return "".join(fakefen) == ChessEnv.replace_tags_board(realfen)


    @staticmethod
    def maybe_flip_fen(fen, flip = False):
        if flip == False:
            return fen
        foo = fen.split(' ')
        rows = foo[0].split('/')
        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        def swapall(aa):
            return "".join([swapcase(a) for a in aa])
        return "/".join( [swapall(row) for row in reversed(rows)] ) \
            + " " + ('w' if foo[1]=='b' else 'b') \
            + " " + "".join( sorted( swapall(foo[2]) ) ) \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]


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
            my_planes = self.to_planes(fen = board_fen)
            history_both.extend(my_planes)
            if len(self.board.move_stack) > 0:
                history_moves.append(self.board.pop())

        for mov in reversed(history_moves):
            self.board.push(mov)
            
        history_both = np.asarray(history_both)
        assert history_both.shape == (96, 8, 8)
        return history_both


    @staticmethod
    def to_planes(fen):
        board_state = ChessEnv.replace_tags_board(fen)
        pieces_both = np.zeros(shape = (12, 8, 8))
        for rank in range(8):
            for file in range(8):
                v = board_state[rank * 8 + file]
                if v.isalpha():
                    pieces_both[ChessEnv.ind[v]][rank][file] = 1
        # pieces_p1 = [ChessEnv.one_hot[val] if val.isupper() else [0, 0, 0, 0, 0, 0] \
        #     for val in board_state]
        # pieces_p1 = np.transpose(np.reshape(pieces_p1, (8, 8, 6)), (2, 0, 1))
        # pieces_p2 = [ChessEnv.one_hot[val] if val.islower() else [0, 0, 0, 0, 0, 0] \
        #     for val in board_state]
        # pieces_p2 = np.transpose(np.reshape(pieces_p2, (8, 8, 6)), (2, 0, 1))
        # assert pieces_p1.shape == (6, 8, 8)
        # state = np.vstack((pieces_p1, pieces_p2))
        assert pieces_both.shape == (12, 8, 8)
        return pieces_both
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

ChessEnv.ind = {ChessEnv.plane_order[i]: i for i in range(12)}