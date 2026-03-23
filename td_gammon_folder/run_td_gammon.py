"""
Main entry point for running TD-Gammon-style experiments.

It simply:
1. creates a configuration object
2. starts training
3. saves CSV results for graphing

All of the actual game logic, neural network logic, and TD(lambda)
learning logic live in the other files.
"""

from td_config import Config
from td_training import train_td_gammon


if __name__ == "__main__":
    cfg = Config()
    train_td_gammon(cfg)