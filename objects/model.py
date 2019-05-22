from env import CollectGoodObjectsEnv

STAGE_ENV_NAME = {
    1: ["cake_hat_true", "balloon_can_true", "cake_can_false", "balloon_hat_false"],
    2: ["train"],
    3: ["test"],
    4: ["hat_false", "cake_false", "balloon_true", "can_true"],
    5: ["hat_true", "cake_true", "balloon_false", "can_false"]
}

class Model(object):
    """The complete model"""
    def __init__(self,
                 stage=1,
                 initialize=False):
        self.env = None

        self.model_v = None
        self.model_m = None
        self.model_c = None


        self.env = self.initialize_env(stage)

    def initialize_env(self, stage):
        return CollectGoodObjectsEnv(name=STAGE_ENV_NAME[stage])
