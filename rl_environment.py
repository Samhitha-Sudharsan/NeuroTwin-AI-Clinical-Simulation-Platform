import gymnasium as gym
from gymnasium import spaces
import numpy as np
from neural_mass_model import JansenRitModel

class NeuroTwinEnv(gym.Env):
    """
    A Custom Gymnasium Environment for training an RL agent to 
    rehabilitate a maladaptive Neural Mass Model.
    """
    def __init__(self):
        super(NeuroTwinEnv, self).__init__()
        
        # We initialize the model in a "maladaptive" state (e.g. FND / Seizure)
        # by tweaking the Excitatory/Inhibitory gains.
        self.model = JansenRitModel(A=4.5, B=10.0) 
        
        # Action space: The external intervention applied to the brain region
        # e.g., Neuromodulation (TMS) or Behavioral Therapy Input
        # Continuous value between -50 and 50
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        
        # Observation space: The 6 internal state variables of the ODE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.max_steps = 1000  # 1 second at dt=0.001
        self.current_step = 0

    def step(self, action):
        intervention = action[0]
        
        # Apply the intervention and advance the simulation by one step
        eeg_output = self.model.step(intervention=intervention)
        self.current_step += 1
        
        # Observation is the internal state of the NMM
        obs = np.copy(self.model.state).astype(np.float32)
        
        # Reward Function: 
        # A maladaptive/seizure state has wild, high-amplitude swings.
        # A healthy state has bounded, low-amplitude alpha/beta waves.
        # We heavily penalize large voltage swings (forcing the AI to stabilize the system).
        # We also slightly penalize massive interventions to encourage subtle therapy.
        reward = -np.abs(eeg_output) - 0.01 * np.abs(intervention)
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # We terminate early if the system blows up completely (failsafe)
        if np.any(np.isnan(obs)) or np.any(np.abs(obs) > 1000):
            reward = -1000.0
            terminated = True
            
        info = {"eeg_output": eeg_output}
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the NMM to baseline and set seed
        self.model.state = np.zeros(6)
        if seed is not None:
            self.model.rng = np.random.default_rng(seed)
        self.current_step = 0
        
        obs = np.copy(self.model.state).astype(np.float32)
        info = {}
        
        return obs, info

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    
    # Test the environment compliance with Gymnasium API
    env = NeuroTwinEnv()
    check_env(env)
    print("NeuroTwinEnv has passed the Gymnasium environment check!")
    
    # Run a random agent for 1 episode
    obs, info = env.reset()
    total_reward = 0
    for _ in range(1000):
        # Random intervention
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
            
    print(f"Random Agent Total Reward: {total_reward:.2f}")
