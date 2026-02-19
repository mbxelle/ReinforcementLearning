# -----------------------------
# VALUE ITERATION

import time

gamma = 0.95
theta = 0.001

size = 4
num_states = size*size
terminals = [0,15]

actions = ["up","down","right","left"]
arrow = {"up":"↑","down":"↓","right":"→","left":"←"}


def to_row_col(s):
    return s//size, s%size


def to_state(r,c):
    return r*size+c


def move(s,a):

    r,c = to_row_col(s)

    if a=="up": r-=1
    elif a=="down": r+=1
    elif a=="right": c+=1
    else: c-=1

    # off-grid means no movement
    if r<0 or r>=size or c<0 or c>=size:
        return s

    return to_state(r,c)


def side_actions(a):
    if a in ["up","down"]:
        return ["left","right"]
    return ["up","down"]


def transitions(s,a,p1,p2,r_a):

    if s in terminals:
        return []

    side_prob=(1-p1-p2)/2

    desired=move(s,a)
    s1,s2=side_actions(a)

    # I realized my original transition model didn't match the assignment FAQ examples (like A10).
    # The key idea is:
    # - If the intended move is inside the grid, the "sideways slip" states are based on the *desired next state*
    #   (the state you tried to go to), not based on the current state.
    # - Also, if a sideways slip from that desired state would go off-grid, that probability gets added back
    #   to the desired state (this is why in A10 p1 + (1-p1-p2)/2 going to the intended state).
    #
    # If the intended move is off-grid, keep the old behavior:
    # stay in place with probability p1+p2, and slip to perpendicular neighbors of the current state.

    out=[]

    if desired==s:
        # intended direction was off-grid -> "no move"
        out.append((p1+p2,s,r_a))
        out.append((side_prob,move(s,s1),r_a))
        out.append((side_prob,move(s,s2),r_a))
    else:
        # intended direction is valid
        out.append((p1,desired,r_a))
        out.append((p2,s,r_a))

        # sideways outcomes are computed from the DESIRED state
        side1_from_desired = move(desired, s1)
        side2_from_desired = move(desired, s2)

        # if a sideways move from desired goes off-grid, move() returns desired,
        # so the probability automatically gets added to the desired state when we merge.
        out.append((side_prob, side1_from_desired, r_a))
        out.append((side_prob, side2_from_desired, r_a))

    # because we might create duplicate next states (like desired showing up twice),
    # we should merge probabilities so they sum correctly.
    merged = {}
    for prob, nxt, r in out:
        # reward r is always r_a (depends on action), so we can just sum probs by nxt
        merged[nxt] = merged.get(nxt, 0) + prob

    final_out = []
    for nxt, prob in merged.items():
        final_out.append((prob, nxt, r_a))

    return final_out


def expected_return(s,a,V,p1,p2,rewards):

    total=0

    for prob,nxt,r in transitions(s,a,p1,p2,rewards[a]):
        total += prob*(r + gamma*V[nxt])

    return total


def print_policy(policy):

    for r in range(size):
        row=[]
        for c in range(size):
            s=to_state(r,c)
            if s in terminals:
                row.append("T")
            else:
                row.append(arrow[policy[s]])
        print(" ".join(row))
    print()


# ==============================
# MAIN VALUE ITERATION
# ==============================

def main():

    p1=float(input("Enter p1: "))
    p2=float(input("Enter p2: "))
    r_up=float(input("Enter r_up: "))
    r_down=float(input("Enter r_down: "))
    r_right=float(input("Enter r_right: "))
    r_left=float(input("Enter r_left: "))


    # (if p1+p2 > 1 then side_prob becomes negative which makes no sense)
    if p1 < 0 or p2 < 0 or p1 + p2 > 1:
        print("Invalid input: need p1>=0, p2>=0, and p1+p2<=1")
        return

    rewards={
        "up":r_up,
        "down":r_down,
        "right":r_right,
        "left":r_left
    }

    # start with V(s)=0
    V=[0.0]*num_states

    times=[]
    it=0

    while True:

        it+=1
        start=time.perf_counter()

        delta=0
        newV=V[:]

        for s in range(num_states):

            if s in terminals:
                continue

            old=V[s]

            # value iteration core:
            # choose best action immediately
            q=[]
            for a in actions:
                q.append(expected_return(s,a,V,p1,p2,rewards))

            newV[s]=max(q)

            delta=max(delta,abs(old-newV[s]))

        V=newV

        end=time.perf_counter()
        times.append(end-start)

        print("Value Iter",it,"delta",delta)

        if delta < theta:
            break

    # extract final greedy policy
    policy=["up"]*num_states

    for s in range(num_states):

        if s in terminals:
            continue

        q=[]
        for a in actions:
            q.append(expected_return(s,a,V,p1,p2,rewards))

        policy[s]=actions[q.index(max(q))]

    print("\nFINAL VALUE ITERATION RESULT")
    print("iterations:",it)
    print_policy(policy)


if __name__=="__main__":
    main()
