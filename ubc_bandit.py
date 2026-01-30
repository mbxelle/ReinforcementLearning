# UCB (Upper Confidence Bound) for a 10-armed Bernoulli bandit
#REFER TO NOTES FOR THE BREAKDOWN OF CODE
import random
import math
import time


# PART A: ENVIRONMENT SETUP (THE SLOT MACHINES)
def make_environment(k=10):
    q = []
    for i in range(k):
        q.append(random.random())  # random probability in [0, 1)
    return q


# PART B: ENVIRONMENT RESPONSE (PULLING AN ARM)
def pull_arm(q, arm):
    r = random.random()           # random number in [0, 1)
    if r < q[arm]:
        return 1
    else:
        return 0


# PART C: RUN ONE UCB EXPERIMENT ON ONE ENVIRONMENT
def run_ucb_one_env(q, rounds=5000, c=2.0, report_every=100):
    k = len(q)

    # PART C1: TRACKING VARIABLES FOR LEARNING
    N = [0] * k
    Q = [0.0] * k

    # PART C2: TRACKING PERFORMANCE (FOR ASSIGNMENT OUTPUT)
    # total_reward tracks the total successes so far.
    total_reward = 0
    # optimal_arm is the arm with the highest true success probability.
    optimal_arm = 0
    for i in range(k):
        if q[i] > q[optimal_arm]:
            optimal_arm = i
    # optimal_chosen counts how many times we picked that best arm.
    optimal_chosen = 0

    # PART C3: INITIALIZATION (PULL EACH ARM ONCE)
    # We do this so that N[a] is not zero and the UCB bonus does not divide by zero.
    t = 0
    for a in range(k):
        reward = pull_arm(q, a)
        N[a] = 1
        Q[a] = float(reward)
        total_reward += reward
        if a == optimal_arm:
            optimal_chosen += 1

        t += 1
        if t % report_every == 0:
            avg_reward = total_reward / t
            print("Round:", t,
                  " optimal chosen:", optimal_chosen,
                  " average reward:", round(avg_reward, 4))

    # PART C4: MAIN UCB LOOP (CHOOSE ARMS USING UCB SCORE)
    # At each round, we compute: UCB(a) = Q[a] + c * sqrt( ln(t) / N[a] )
    # AND THen we choose the arm with the largest UCB score.
    while t < rounds:
        ucb_scores = []

        for a in range(k):
            bonus = c * math.sqrt(math.log(t + 1) / N[a])  # bonus is bigger if N[a] is small because we want to explore more
            score = Q[a] + bonus # total UCB score
            ucb_scores.append(score)

        chosen_arm = ucb_scores.index(max(ucb_scores))

        # PART C5: TAKE ACTION AND OBSERVE REWARD (0 OR 1)
        reward = pull_arm(q, chosen_arm)
        total_reward += reward
        if chosen_arm == optimal_arm:
            optimal_chosen += 1

        # PART C6: UPDATE COUNTS AND UPDATE THE AVERAGE (SAMPLE-AVERAGE UPDATE)
        N[chosen_arm] += 1
        Q[chosen_arm] = Q[chosen_arm] + (reward - Q[chosen_arm]) / N[chosen_arm]

        t += 1
        if t % report_every == 0:
            avg_reward = total_reward / t
            print("Round:", t,
                  " optimal chosen:", optimal_chosen,
                  " average reward:", round(avg_reward, 4))

    # Return results in case you want to summarize across runs
    return {
        "optimal_arm": optimal_arm,
        "optimal_chosen": optimal_chosen,
        "avg_reward": total_reward / rounds
    }


# PART D: RUN 100 TIMES WITH 100 DIFFERENT ENVIRONMENTS
# runs UCB for 5000 rounds.
def run_100_experiments(num_runs=100, rounds=5000, c=2.0, report_every=100):
    all_avg_rewards = []
    all_optimal_rates = []

    for run in range(1, num_runs + 1):
        print("Starting run", run, "out of", num_runs)

        q = make_environment(k=10)
        result = run_ucb_one_env(q, rounds=rounds, c=c, report_every=report_every)

        all_avg_rewards.append(result["avg_reward"])
        all_optimal_rates.append(result["optimal_chosen"] / rounds)

    # PART E: PRINT A SIMPLE SUMMARY OVER 100 RUNS
    mean_avg_reward = sum(all_avg_rewards) / len(all_avg_rewards)
    mean_optimal_rate = sum(all_optimal_rates) / len(all_optimal_rates)
    print("SUMMARY OVER", num_runs, "RUNS")
    print("Mean average reward:", round(mean_avg_reward, 4))
    print("Mean optimal-action rate:", round(mean_optimal_rate, 4))


# PART F: PROGRAM ENTRY POINT
if __name__ == "__main__":
    random.seed(int(time.time()))   # time-based seed (note: this is similar idea to srand(time(NULL)) in C)

    #at least 5000 rounds, print every 100 rounds, run 100 environments.
    run_100_experiments(num_runs=100, rounds=5000, c=2.0, report_every=100)
