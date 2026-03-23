"""
Configuration values for the TD-Gammon-style implementation.

Keeping these in a separate file makes it easier to:
- tune experiments
- explain hyperparameters
- keep the runner file short
"""

from dataclasses import dataclass


@dataclass
class Config:
    """
    Stores experiment settings.

    These are the values you will most likely tweak when collecting data.

    Main groups:
    - randomness / reproducibility
    - neural network size
    - TD(lambda) hyperparameters
    - training schedule
    - evaluation schedule
    - output file paths
    """

    seed: int = 7

    # Neural network / TD(lambda) settings
    hidden_size: int = 128
    alpha: float = 0.0015
    lam: float = 0.7
    gamma: float = 1.0

    # Training schedule
    episodes: int = 300
    eval_interval: int = 25
    eval_games: int = 20
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    max_turns_per_game: int = 300

    # Search settings
    use_two_ply_search: bool = False
    two_ply_sample_rolls: int = 4

    # Output files for graphing
    metrics_csv: str = "full_td_gammon_metrics.csv"
    eval_csv: str = "full_td_gammon_eval.csv"