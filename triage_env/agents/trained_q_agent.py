from triage_env.agents.q_learning_agents import QLearningAgent


class TrainedQAgent(QLearningAgent):
    def __init__(self, model_path):
        super().__init__(alpha=0.0, gamma=0.95, epsilon=0.0)
        self.load(model_path)