"""Trainer module for RL agents in Medical Triage environment."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Trainer:
    """Base trainer for RL agents."""
    
    def __init__(self, agent: Any, env: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer.
        
        Args:
            agent: RL agent to train
            env: Environment instance
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config or {}
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Training metrics dictionary
        """
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < 1000:
                action = self.agent.act(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Update agent
                if hasattr(self.agent, 'update'):
                    self.agent.update(reward, done, info)
                
                episode_reward += reward
                episode_length += 1
                done = done or truncated
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
        
        return {
            "mean_reward": float(np.mean(self.episode_rewards[-100:])),
            "mean_length": float(np.mean(self.episode_lengths[-100:])),
            "total_episodes": num_episodes
        }
