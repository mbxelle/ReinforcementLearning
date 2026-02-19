# -----------------------------
# POLICY ITERATION - Gridworld


import time

# assignment constants
gamma = 0.95   # future reward discount
theta = 0.001  # stopping condition for value updates

size = 4
num_states = size * size

# Example 4.1 terminal states
terminals = [0, 15]

actions = ["up", "down", "right", "left"]
arrow = {"up":"↑", "down":"↓", "right":"→", "left":"←"}


# ----- helper conversions -----

def to_row_col(s):
    # convert state number into row/col so movement is easier
    return s // size, s % size


def to_state(r, c):
    # convert row/col back to single state index
    return r * size + c


# ----- movement -----

def move(s, a):
    # tries to move agent in chosen direction
    # if move goes outside grid -> stays same (book rule)

    r, c = to_row_col(s)

    if a == "up":
        r -= 1
    elif a == "down":
        r += 1
    elif a == "right":
        c += 1
    else:
        c -= 1

    if r < 0 or r >= size or c < 0 or c >= size:
        return s

    return to_state(r, c)


def side_actions(a):
    # sideways actions used in stochastic transition
    if a in ["up","down"]:
        return ["left","right"]
    return ["up","down"]


# ----- transition model -----

def transitions(s, a, p1, p2, r_a):
    """
    Assignment stochastic rule:
    p1 -> intended direction
    p2 -> stay in place
    remaining probability split sideways
    """

    if s in terminals:
        return []

    side_prob = (1 - p1 - p2) / 2

    desired = move(s, a)
    s1, s2 = side_actions(a)

    out = []

    # if intended move hits wall, that probability becomes "stay"
    if desired == s:
        out.append((p1 + p2, s, r_a))

        # sideways outcomes (from current state)
        out.append((side_prob, move(s, s1), r_a))
        out.append((side_prob, move(s, s2), r_a))

    else:
        out.append((p1, desired, r_a))
        out.append((p2, s, r_a))

        # sideways outcomes are computed from the DESIRED state
        out.append((side_prob, move(desired, s1), r_a))
        out.append((side_prob, move(desired, s2), r_a))

    # merge duplicates so probabilities add up correctly
    merged = {}
    for prob, nxt, r in out:
        merged[nxt] = merged.get(nxt, 0) + prob

    final_out = []
    for nxt, prob in merged.items():
        final_out.append((prob, nxt, r_a))

    return final_out


# ----- bbellman expectation -----

def expected_return(s, a, V, p1, p2, rewards):
    # computes expected future return for one action
    total = 0

    for prob, nxt, r in transitions(s, a, p1, p2, rewards[a]):
        total += prob * (r + gamma * V[nxt])

    return total


# ----- print policy nicely -----

def print_policy(policy):

    for r in range(size):
        row = []
        for c in range(size):

            s = to_state(r,c)

            if s in terminals:
                row.append("T")
            else:
                row.append(arrow[policy[s]])

        print(" ".join(row))
    print()


# =================================================
# MAIN POLICY ITERATION
# =================================================

def main():

    # required user inputs from assignment
    p1 = float(input("Enter p1: "))
    p2 = float(input("Enter p2: "))
    r_up = float(input("Enter r_up: "))
    r_down = float(input("Enter r_down: "))
    r_right = float(input("Enter r_right: "))
    r_left = float(input("Enter r_left: "))

    rewards = {
        "up": r_up,
        "down": r_down,
        "right": r_right,
        "left": r_left
    }

    # initialize value function to zero
    V = [0.0 for _ in range(num_states)]

    # equiprobable random policy (0.25 each action)
    policy_probs = []
    for s in range(num_states):
        if s in terminals:
            policy_probs.append([0,0,0,0])
        else:
            policy_probs.append([0.25,0.25,0.25,0.25])

    policy = ["up"] * num_states

    outer_iter = 0
    times = []

    while True:

        outer_iter += 1
        start = time.perf_counter()

        # ======================
        # POLICY EVALUATION
        # evaluate current policy until values stabilize
        # ======================
        while True:

            delta = 0

            for s in range(num_states):

                if s in terminals:
                    continue

                old = V[s]
                new = 0

                # weighted average over actions
                for i,a in enumerate(actions):
                    new += policy_probs[s][i] * expected_return(
                        s,a,V,p1,p2,rewards
                    )

                V[s] = new
                delta = max(delta, abs(old-new))

            if delta < theta:
                break


        # ======================
        # POLICY IMPROVEMENT
        # make policy greedy using current values
        # ======================
        stable = True

        for s in range(num_states):

            if s in terminals:
                continue

            old_best = max(range(4),
                           key=lambda i: policy_probs[s][i])

            q = []
            for a in actions:
                q.append(expected_return(s,a,V,p1,p2,rewards))

            new_best = q.index(max(q))
            policy[s] = actions[new_best]

            # deterministic policy update
            for i in range(4):
                policy_probs[s][i] = 1 if i==new_best else 0

            if new_best != old_best:
                stable = False

        end = time.perf_counter()
        times.append(end-start)

        print("Policy Iteration", outer_iter,
              "time:", times[-1])

        if stable:
            break

    print("\nFINAL POLICY ITERATION RESULT")
    print("iterations:", outer_iter)
    print_policy(policy)


if __name__ == "__main__":
    main()
