class RosDefaultInitialState:
    def __init__(self, env):
        self.env = env

    def step(self, obs, reward, absorbing, info):
        return

    def setup(self):
        qpos, qvel, qacc, _, _ = self.env.connection.get_data()
        
        return qpos, qvel, qacc
