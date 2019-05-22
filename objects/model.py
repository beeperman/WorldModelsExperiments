from env import CollectGoodObjectsTrainEnv


class Model(object):
    """The complete model"""
    def __init__(self,
                 initialize=False):
        self.env = None

        self.model_v = None
        self.model_m = None
        self.model_c = None


        self.env = self.initialize_env()

    def initialize_env(self):
        return CollectGoodObjectsTrainEnv()
