import os
import chess

def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")


def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']
    pairs = []
    pairs_promoted = []
    for l in letters:
        for n in numbers:
            pairs.append(l + n)
            if n == '1' or n == '8':
                for p in promoted_to:
                    pairs_promoted.append(l + n + p)

    for p1 in pairs:
        for p2 in pairs:
            if p1 != p2:
                try:
                    _ = chess.Move.from_uci(p1 + p2)
                    labels_array.append(p1 + p2)
                except:
                    pass

    for p1 in pairs:
        for pro in pairs_promoted:
            try:
                _ = chess.Move.from_uci(p1 + pro)
                labels_array.append(p1 + pro)
            except:
                pass

    return labels_array


class Config:
    def __init__(self, config_type="mini"):
        self.opts = Options()
        self.resource = ResourceConfig()

        if config_type == "mini":
            import chess_zero.configs.mini as c
        elif config_type == "normal":
            import chess_zero.configs.normal as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()
        self.labels = create_uci_labels()
        self.n_labels = len(self.labels)


class Options:
    new = False


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class PlayWithHumanConfig:
    def __init__(self):
        self.simulation_num_per_move = 2
        self.thinking_loop = 1
        self.logging_thinking = True
        self.c_puct = 5
        self.parallel_search_num = 2
        self.noise_eps = 0
        self.change_tau_turn = 0
        self.resign_threshold = None

    def update_play_config(self, pc):
        """

        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.thinking_loop = self.thinking_loop
        pc.logging_thinking = self.logging_thinking
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.change_tau_turn = self.change_tau_turn
        pc.parallel_search_num = self.parallel_search_num
        pc.resign_threshold = self.resign_threshold
