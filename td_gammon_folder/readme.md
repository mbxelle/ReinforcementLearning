README

Project name:
TD-Gammon-Style Backgammon (NumPy version)

What this project is

This is a TD-Gammon-style backgammon project.
It uses:
- self-play
- a neural network value function
- TD(lambda) learning
- afterstate evaluation

It uses NumPy instead, so it can run on systems where torch is not installed.

--------------------------------------------------

IMPORTANT (REQUIRED PACKAGE)

--------------------------------------------------

You must install NumPy before running the code.

Run:

python3 -m pip install numpy

If pip is missing:

python3 -m ensurepip --upgrade
python3 -m pip install numpy

--------------------------------------------------
FILES AND WHAT THEY CONTAIN
--------------------------------------------------

run_td_gammon.py
- This is the main file you run
- It starts the whole experiment
- It loads the config and begins training

td_config.py
- This stores all settings
- Number of episodes
- Learning rate
- Lambda
- Gamma
- Hidden size
- Epsilon settings
- Search settings
- CSV output file names

td_model.py
- This contains the neural network
- The network predicts how good a board position is
- It learns a value function V(s)
- This version uses NumPy, not torch

td_env.py
- This contains the backgammon game environment
- It stores the board
- It handles rules and move generation
- It also encodes the board into features for the neural network

td_utils.py
- Small helper functions
- Random seed setup
- Epsilon decay schedule

td_search.py
- This contains move selection
- It generates legal afterstates
- It scores them with the neural network
- It picks the best move
- It can also do optional 2-ply search

td_learning.py
- This contains the TD(lambda) update
- This is where the neural network weights are updated
- It uses eligibility traces

td_eval.py
- This contains evaluation code
- It tests the learned agent against a random opponent
- It computes win rate and average turns

td_training.py
- This contains the main training loop
- It connects everything together:
  environment + search + TD update + evaluation + CSV logging

--------------------------------------------------
WHAT THIS VERSION OF TD BACKGAMMON CONTAINS
--------------------------------------------------

This version contains:
- full 24-point backgammon board
- 15 checkers per player
- two dice per turn
- doubles as four moves
- hits and blots
- closed points / blockades
- bar and re-entry
- bearing off
- oversized bear-off rules
- full-turn legal move generation
- afterstate evaluation
- neural network value function
- self-play learning
- TD(lambda) with eligibility traces
- optional 2-ply search
- terminal rewards for:
  - normal win
  - gammon
  - backgammon
- CSV result logging for graphing

--------------------------------------------------
WHAT THIS VERSION DOES NOT CONTAIN
--------------------------------------------------

This version does not include:
- doubling cube
- take / pass cube decisions
- match-play rules
- Crawford rule
- match equity tables
- deep multi-ply search beyond the shallow setting
- the exact original historical TD-Gammon architecture
- the extremely large training scale of the original system

--------------------------------------------------
HOW TO RUN IT
--------------------------------------------------

1. Put all files in the same folder

2. Open terminal in that folder

3. Run:

python3 run_td_gammon.py

--------------------------------------------------
WHAT SHOULD HAPPEN
--------------------------------------------------

If everything works, you should see:
- training start message
- episode information
- evaluation output every few episodes
- final message saying training is complete
- CSV files saved

--------------------------------------------------
OUTPUT FILES
--------------------------------------------------

The project should produce these result files:

full_td_gammon_metrics.csv
This contains training metrics such as:
- episode
- epsilon
- turns
- network updates
- winner
- pip counts
- average absolute TD error

full_td_gammon_eval.csv
This contains evaluation metrics such as:
- episode
- win rate vs random
- loss rate vs random
- average turns in evaluation games

--------------------------------------------------
QUICK TEST SETTINGS
--------------------------------------------------

For a quick test, use smaller values in td_config.py:

episodes = 10
eval_interval = 5
eval_games = 4
use_two_ply_search = False

Then run:

python3 run_td_gammon.py

If that works, your file split is correct.

--------------------------------------------------
RECOMMENDED EXPERIMENT RUNS (FOR GOOD DATA)
--------------------------------------------------

To get meaningful and reliable results, the experiment should be run
multiple times and with a larger number of training episodes.

Recommended approach:

1. Small test run (debugging)
- episodes = 10–20
- Purpose: verify that the code runs correctly

2. Medium run (basic results)
- episodes = 200–300
- Purpose: observe initial learning trends

3. Full experiment (good data)
- episodes = 1000+ (recommended)
- eval_games = 20–50
- Purpose: produce smoother and more reliable graphs

4. Multiple runs (important for accuracy)
- Run the same configuration 2–3 times with different seeds
- This helps reduce randomness from dice rolls and initialization

Why multiple runs are important:
- Backgammon is stochastic (dice-based)
- Neural network training is also random
- Results from a single run may be noisy or misleading

Best practice:
Report average trends across multiple runs, not just one.

Example:
- Run 3 experiments with episodes = 1000
- Compare or average the win_rate_vs_random curves

--------------------------------------------------