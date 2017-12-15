from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

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

QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


class ChessPlayer:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ChessModelAPI(self.config, self.model)

        self.move_lookup = {k:v for k,v in zip((chess.Move.from_uci(move) for move in self.config.labels),range(len(self.config.labels)))}
        self.labels_n = config.n_labels
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun
        self.reset()

    # we are leaking memory + losing MCTS nodes...!! (without this)
    def reset(self):
        # these are from AGZ nature paper
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,))) # dict: stateKey->int
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_u = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.expanded = set()
        self.now_expanding = set()

    def sl_action(self, board, action):

        env = ChessEnv().update(board)

        policy = np.zeros(self.labels_n)
        k = 0
        for mov in self.config.labels:
            if mov == action:
                policy[k] = 1.0
                break
            k += 1

        self.moves.append([env.observation, list(policy)])
        return action

    def action(self, env, can_stop = True):
        self.reset()

        self.my_color = env.board.turn

        state = self.state_key(env)

        for tl in range(self.play_config.thinking_loop):
            self.search_moves(env)
            policy = self.calc_policy(env)
            action = int(np.random.choice(range(self.labels_n), p = policy))
            action_by_value = int(np.argmax(self.var_q[state] + (self.var_n[state] > 0)*100))
            #print(len(self.expanded))
            #assert len(self.expanded) == (tl+1)*self.play_config.simulation_num_per_move
            if action == action_by_value or env.turn < self.play_config.change_tau_turn:
                break
            # if tl > 0 and self.play_config.logging_thinking:
            #     uci.info(depth = tl+1,move=self.config.labels[action],score=self.var_q[state][action])
                # logger.debug(f"continue thinking: policy move=({action % 8}, {action // 8}), "
                #              f"value move=({action_by_value % 8}, {action_by_value // 8})")

        # this is for play_gui, not necessary when training.
        self.thinking_history[env.observation] = HistoryItem(action, policy, list(self.var_q[state]), list(self.var_n[state]))

        if can_stop and self.play_config.resign_threshold is not None and \
                        np.max(self.var_q[state] - (self.var_n[state] == 0) * 10) <= self.play_config.resign_threshold \
                        and self.play_config.min_resign_turn < env.turn:
            return None
        elif can_stop and env.turn >= self.play_config.average_chess_movements:
            env.ending_average_game()
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[action]

    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    @profile
    def search_moves(self, env):
        start = time.time()
        self.running_simulation_num = 0

        coroutine_list = [ self.start_search_my_move(env) \
            for _ in range(self.play_config.simulation_num_per_move) ]

        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))
        #logger.debug(f"Search time per move: {time.time()-start}")
        # uncomment to see profile result per move
        # raise

    async def start_search_my_move(self, env) -> float:
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            my_env = env.copy()  # use this option to preserve history
            leaf_v = await self.search_my_move(my_env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
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

        while state in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if state not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env.copy())
            return leaf_v # I'm returning everything from the POV of side to move

        action_t = self.select_action_q_and_u(env, is_root_node)

        env.step(self.config.labels[action_t])

        virtual_loss = self.config.play.virtual_loss
        self.var_n[state][action_t] += virtual_loss
        self.var_w[state][action_t] -= virtual_loss

        leaf_v = await self.search_my_move(env)  # next move
        leaf_v = -leaf_v # from enemy POV

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q, U
        n = self.var_n[state][action_t] = self.var_n[state][action_t] - virtual_loss + 1
        w = self.var_w[state][action_t] = self.var_w[state][action_t] + virtual_loss + leaf_v
        self.var_q[state][action_t] = w / n

        return leaf_v

    @profile
    async def expand_and_evaluate(self, env) -> float:
        """expand new leaf

        insert var_p[state], return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        state = self.state_key(env)
        self.now_expanding.add(state)

        state_planes = env.canonical_input_planes()

        future = await self.predict(state_planes)  # type: Future

        await future
        leaf_p, leaf_v = future.result() # these are canonical policy and value (i.e. side to move is "white")

        if env.board.turn == chess.BLACK:
            leaf_p = Config.flip_policy(leaf_p) # get it back to python-chess form

        self.var_p[state] = leaf_p  # P is policy for next_player (black or white)

        self.expanded.add(state)
        self.now_expanding.remove(state)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            #logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        """
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

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

    @staticmethod
    def state_key(env: ChessEnv):
        fen = env.board.fen().rsplit(' ',1) # drop the halfmove clock
        return fen[0]

    def select_action_q_and_u(self, env, is_root_node) -> int:
        state = self.state_key(env)

        """Bottlenecks are these two lines"""
        legal_moves = [self.move_lookup[mov] for mov in env.board.legal_moves]
        legal_labels = np.zeros(len(self.config.labels))
        #logger.debug(legal_moves)
        legal_labels[legal_moves] = 1

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

        # if env.board.turn == self.my_color:
        v_ = (self.var_q[state] + u_ + 1000) * legal_labels
        # else:
        #     # When enemy's selecting action, flip Q-Value.
        #     v_ = (-self.var_q[state] + u_ + 1000) * legal_labels

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t
