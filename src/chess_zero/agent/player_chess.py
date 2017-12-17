from concurrent.futures import Future, ThreadPoolExecutor
from collections import defaultdict, namedtuple
from logging import getLogger
from threading import Thread,Lock

from profilehooks import profile

import time

import numpy as np
import chess

from chess_zero.agent.api_chess import ChessModelAPI
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
#from chess_zero.play_game.uci import info

import platform
if platform.system() != "Windows":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

QueueItem = namedtuple("QueueItem", "state v future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


class ChessPlayer:
    dot = False
    def __init__(self, config: Config, model=None, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ChessModelAPI(self.config, self.model)

        self.move_lookup = {k:v for k,v in zip((chess.Move.from_uci(move) for move in self.config.labels),range(len(self.config.labels)))}
        self.labels_n = config.n_labels
        self.labels = config.labels
        self.prediction_queue_lock = Lock()
        self.search_threads = ThreadPoolExecutor(max_workers=self.play_config.parallel_search_num)
        self.is_thinking = False

        self.moves = []

        self.thinking_history = {}  # for fun
        self.reset()

    # we are leaking memory + losing MCTS nodes...!! (without this)
    def reset(self):
        # these are from AGZ nature paper
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,))) # visit count
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,))) # total action value
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,))) # mean action value
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,))) # prior probability

        self.legal_binary = defaultdict()
        self.visited = set()
        self.node_lock = defaultdict(Lock)
        self.prediction_queue = []

    def sl_action(self, board, action):

        env = ChessEnv().update(board)

        policy = np.zeros(self.labels_n)
        k = self.move_lookup[chess.Move.from_uci(action)] 
        policy[k] = 1.0

        self.moves.append([env.observation, list(policy)])
        return action

    def deboog(self, env):
        print(env.testeval())
        
        state = self.state_key(env)
        stats = []
        for move in env.board.legal_moves:
            moi = self.move_lookup[move]
            stats.append(np.asarray([self.var_n[state][moi], self.var_w[state][moi], self.var_q[state][moi], self.var_p[state][moi], moi]))
        stats=np.asarray(stats)
        a = stats[stats[:,0].argsort()[::-1]]

        for s in a:
            print(f"{self.labels[int(s[4])]:5}: "
                f"n: {s[0]:3.0f} "
                f"w: {s[1]:7.3f} "
                f"q: {s[2]:7.3f} "
                f"p: {s[3]:7.5f}")

    def action_with_policy(self, env, can_stop = True):
        return self.action(env,can_stop), self.calc_policy(env)

    def action(self, env, can_stop = True):
        self.reset()
        
        state = self.state_key(env)

        self.is_thinking = True
        prediction_worker = Thread(target=self.predict_batch_worker,name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

        for tl in range(self.play_config.thinking_loop):
            self.search_moves(env)
            policy = self.calc_policy(env)
            action = int(np.random.choice(range(self.labels_n), p = policy))
            action_by_value = int(np.argmax(self.var_q[state] + (self.var_n[state] > 0)*100))
            # if env.turn < self.play_config.change_tau_turn:
            #     break
            # if tl > 0 and self.play_config.logging_thinking:
            #     uci.info(depth = tl+1,move=self.config.labels[action],score=self.var_q[state][action])
                # logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), "
                #              f"value move=({action_by_value % 8}, {action_by_value // 8})")
        self.is_thinking = False
        # prediction_worker.join()

        # this is for play_gui, not necessary when training.
        self.thinking_history[env.observation] = HistoryItem(action, policy, list(self.var_q[state]), list(self.var_n[state]))
        #self.deboog(env)
        if can_stop and self.play_config.resign_threshold is not None and \
                        np.max(self.var_q[state] - (self.var_n[state] == 0) * 10) <= self.play_config.resign_threshold \
                        and self.play_config.min_resign_turn < env.turn:
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[action]

    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    #@profile
    def search_moves(self, env):

        if ChessPlayer.dot == False:
            import stacktracer
            stacktracer.trace_start("trace.html")
            ChessPlayer.dot = True

        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.parallel_search_num) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_my_move,env=env.copy(),is_root_node=True))
        [f.result() for f in futures]  

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        """

        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return: leaf value
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            if (env.winner == Winner.white) == (env.board.turn == chess.WHITE):
                return 1 # winner is side-to-move
            return -1

        state = self.state_key(env)

        my_lock = self.node_lock[state]

        with my_lock:
            if state not in self.visited:
                self.visited.add(state)
                #print(state)
                leaf_v = self.expand_and_evaluate(env = env) 
                return leaf_v # I'm returning everything from the POV of side to move


        assert state in self.visited

        # SELECT STEP
        with my_lock:
            action_t = self.select_action_q_and_u(env, is_root_node)

        env.step(self.config.labels[action_t])

        virtual_loss = self.play_config.virtual_loss

        with my_lock:
            self.var_n[state][action_t] += virtual_loss
            self.var_w[state][action_t] -= virtual_loss

        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

            # BACKUP STEP
            # on returning search path
            # update: N, W, Q, U
        with my_lock:
            n = self.var_n[state][action_t] = self.var_n[state][action_t] - virtual_loss + 1
            w = self.var_w[state][action_t] = self.var_w[state][action_t] + virtual_loss + leaf_v
            self.var_q[state][action_t] = w / n

        return leaf_v

    #@profile
    def expand_and_evaluate(self, env) -> float:
        """expand new leaf

        insert var_p[state], return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """

        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(statex=state_planes,testv=env.testeval())
        # these are canonical policy and value (i.e. side to move is "white")

        if env.board.turn == chess.BLACK:
            leaf_p = Config.flip_policy(leaf_p) # get it back to python-chess form

        #np.testing.assert_array_equal(Config.flip_policy(Config.flip_policy(leaf_p)), leaf_p)  

        state = self.state_key(env)

        self.var_p[state] = leaf_p  # P is policy for next_player (black or white)

        return float(leaf_v)

    def predict_batch_worker(self):
        while self.is_thinking:
            if self.prediction_queue:
                with self.prediction_queue_lock:
                    item_list = self.prediction_queue
                    self.prediction_queue = []

                #logger.debug(f"predicting {len(item_list)} items")
                data = np.array([x.state for x in item_list])
                policy_ary, value_ary = self.api.predict(data)
                for item, p, v in zip(item_list, policy_ary, value_ary):
                    item.future.set_result((p, v))
            else:
                time.sleep(self.play_config.prediction_worker_sleep_sec)

    def predict(self, statex, testv):
        future = Future()
        item = QueueItem(statex,testv, future)
        with self.prediction_queue_lock: # lists are atomic anyway though
            self.prediction_queue.append(item)
        return future.result()

    def calc_policy(self, env):
        """calc π(a|s0)
        :return:
        """
        pc = self.play_config

        state = self.state_key(env)
        if env.turn < pc.change_tau_turn:
            return self.var_n[state] / (np.sum(self.var_n[state])+1e-8)  # tau = 1
        else:
            action = np.argmax(self.var_n[state])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    def select_action_q_and_u(self, env, is_root_node) -> int:
        state = self.state_key(env)

        """Bottlenecks are these two lines"""
        if state not in self.legal_binary:
            legal_moves = [self.move_lookup[mov] for mov in env.board.legal_moves]
            legal_labels = np.zeros(len(self.config.labels))
            #logger.debug(legal_moves)
            legal_labels[legal_moves] = 1
            self.legal_binary[state] = legal_labels
        else:
            legal_labels = self.legal_binary[state]

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[state]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[state]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.labels_n)

        # re-normalize in legal moves
        p_ = p_ * legal_labels
        if np.sum(p_) > 0:
            p_ = p_ / np.sum(p_)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[state]) # element-wise division...

        v_ = (self.var_q[state] + u_ + 1000) * legal_labels
        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t

    @staticmethod
    def state_key(env: ChessEnv):
        fen = env.board.fen().rsplit(' ',1) # drop the move clock
        return fen[0]

    def finish_game(self, z):
        """
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]