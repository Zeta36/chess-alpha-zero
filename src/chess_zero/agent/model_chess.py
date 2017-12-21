import hashlib
import json
import urllib.request
import ftplib
import os
from logging import getLogger
# noinspection PyPep8Naming
import keras.backend as k

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from chess_zero.agent.api_chess import ChessModelAPI
from chess_zero.config import Config

logger = getLogger(__name__)


class ChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.queue = None

    def get_api_queue(self):
        if self.queue is None:
            self.queue = ChessModelAPI(self.config, self).prediction_queue
        return self.queue

    def build(self):
        mc = self.config.model
        in_x = x = Input((18, 8, 8))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)
        

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path):
        mc = self.config.model
        resources = self.config.resource
        if mc.distributed and config_path == resources.model_best_config_path:
            try:
                logger.debug("loading model from server")
                ftp_connection = ftplib.FTP(resources.model_best_distributed_ftp_server,
                                            resources.model_best_distributed_ftp_user,
                                            resources.model_best_distributed_ftp_password)
                ftp_connection.cwd(resources.model_best_distributed_ftp_remote_path)
                ftp_connection.retrbinary("RETR model_best_config.json", open(config_path, 'wb').write)
                ftp_connection.retrbinary("RETR model_best_weight.h5", open(weight_path, 'wb').write)
                ftp_connection.quit()
            except:
                pass
        from tensorflow import get_default_graph
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.graph = get_default_graph()
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            #print(self.model.summary)
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

        mc = self.config.model
        resources = self.config.resource
        if mc.distributed and config_path == resources.model_best_config_path:
            try:
                logger.debug("saving model to server")
                ftp_connection = ftplib.FTP(resources.model_best_distributed_ftp_server,
                                            resources.model_best_distributed_ftp_user,
                                            resources.model_best_distributed_ftp_password)
                ftp_connection.cwd(resources.model_best_distributed_ftp_remote_path)
                fh = open(config_path, 'rb')
                ftp_connection.storbinary('STOR model_best_config.json', fh)
                fh.close()

                fh = open(weight_path, 'rb')
                ftp_connection.storbinary('STOR model_best_weight.h5', fh)
                fh.close()
                ftp_connection.quit()
            except:
                pass
