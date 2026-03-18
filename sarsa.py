import random
import time


class GridWorld_SARSA:
    def __init__(self, episodes, alpha=0.1, gamma=0.9, p1=1.0, p2=0.0, epsilon=0.1, epsilonDecay=0.0):
        # probabilities must be valid
        assert p1 + p2 <= 1.0

        self.grid = self.generateGrid()
        self.qTable = self.generateQTable()

        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.p1 = p1
        self.p2 = p2
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay

        self.agentPosition = self.initializeAgentPosition()
        self.manhattanDistance = 1

    # start agent in random valid position
    def initializeAgentPosition(self):
        while True:
            row = random.randint(0, 10)
            col = random.randint(0, 10)

            if self.grid[row][col] == " ":
                self.agentPosition = (row, col)

                # used for normalizing stats
                self.manhattanDistance = row + (11 - col)
                if self.manhattanDistance <= 0:
                    self.manhattanDistance = 1

                return self.agentPosition

    # create 11x11 grid with walls and goal
    def generateGrid(self):
        grid = []

        for i in range(11):
            if i == 0:
                grid.append([" ", " ", " ", " ", " ", "|", " ", " ", " ", " ", "G"])
            elif i == 5:
                grid.append(["-", "-", " ", "-", "-", "+", "-", "-", " ", "-", "-"])
            elif i == 2 or i == 8:
                grid.append([" "] * 11)
            else:
                grid.append([" ", " ", " ", " ", " ", "|", " ", " ", " ", " ", " "])

        return grid

    # initialize Q(s,a) = 0
    def generateQTable(self):
        qTable = {}

        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == " " or self.grid[r][c] == "G":
                    qTable[(r, c)] = [0.0, 0.0, 0.0, 0.0]

        return qTable

    # returns perpendicular actions (used for slipping)
    def adjacentActions(self, action):
        if action in [0, 1]:
            return [2, 3]
        else:
            return [0, 1]

    # epsilon-greedy action selection
    def chooseAction(self, state):
        q_values = self.qTable[state]

        # explore
        if random.random() < self.epsilon:
            preferred = random.choice([0, 1, 2, 3])
        else:
            # exploit (argmax)
            max_q = max(q_values)
            best = [i for i, v in enumerate(q_values) if v == max_q]
            preferred = random.choice(best)

        # environment randomness (p1, p2)
        adj = self.adjacentActions(preferred)
        adjProb = (1 - self.p1 - self.p2) / 2

        move = random.choices(
            [preferred, -1, adj[0], adj[1]],
            weights=[self.p1, self.p2, adjProb, adjProb],
            k=1
        )[0]

        return move

    def inBounds(self, pos):
        r, c = pos
        return 0 <= r <= 10 and 0 <= c <= 10

    def notWall(self, pos):
        r, c = pos
        return self.grid[r][c] == " " or self.grid[r][c] == "G"

    def canUpdatePos(self, pos):
        return self.inBounds(pos) and self.notWall(pos)

    # reward function
    def calculateReward(self, pos):
        if self.grid[pos[0]][pos[1]] == "G":
            return 500
        return -1

    # take a step in environment
    def step(self, move):
        delta = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            -1: (0, 0)   # no move
        }

        dr, dc = delta[move]
        newPos = (self.agentPosition[0] + dr, self.agentPosition[1] + dc)

        if self.canUpdatePos(newPos):
            self.agentPosition = newPos

        reward = self.calculateReward(self.agentPosition)
        done = reward == 500

        return self.agentPosition, reward, done

    # SARSA update: Q(s,a) <- Q + alpha*(r + gamma*Q(s',a') - Q)
    def runEpisode(self):
        state = self.initializeAgentPosition()
        action = random.choice([0, 1, 2, 3])  # first move random
        steps = 0
        done = False

        while not done:
            next_state, reward, done = self.step(action)
            steps += 1

            if done:
                target = reward
                self.qTable[state][action] += self.alpha * (target - self.qTable[state][action])
            else:
                next_action = self.chooseAction(next_state)

                target = reward + self.gamma * self.qTable[next_state][next_action]
                self.qTable[state][action] += self.alpha * (target - self.qTable[state][action])

                state = next_state
                action = next_action

        return steps

    # print final policy using arrows
    def visualizePolicy(self):
        arrows = {0: "^", 1: "v", 2: "<", 3: ">"}

        print("\n--- Learned Policy ---")

        for r in range(len(self.grid)):
            row_out = []

            for c in range(len(self.grid[r])):
                cell = self.grid[r][c]

                if cell == "G":
                    row_out.append("G")
                elif cell in ["|", "-", "+"]:
                    row_out.append(cell)
                else:
                    q = self.qTable[(r, c)]

                    if q == [0.0, 0.0, 0.0, 0.0]:
                        row_out.append("?")
                    else:
                        best = q.index(max(q))
                        row_out.append(arrows[best])

            print(" ".join(row_out))

        print()

    # run all episodes and track stats
    def runEpisodes(self):
        print(f"SARSA (alpha={self.alpha}, epsilon={self.epsilon})")

        totalSteps = 0
        totalTime = 0

        for i in range(1, self.episodes + 1):
            start = time.time()

            steps = self.runEpisode()

            end = time.time()

            totalSteps += steps
            totalTime += (end - start)

            if i % 500 == 0:
                print(f"Episode {i}: avg steps = {totalSteps/i:.2f}")

        print("\nFinal Stats:")
        print("Episodes:", self.episodes)
        print("Total Steps:", totalSteps)
        print("Total Time:", totalTime)


if __name__ == "__main__":
    p1 = float(input("Enter p1: "))
    p2 = float(input("Enter p2: "))

    # test epsilon = 0.05
    sarsa_e005 = GridWorld_SARSA(episodes=10000, alpha=0.1, epsilon=0.05, p1=p1, p2=p2)
    sarsa_e005.runEpisodes()
    sarsa_e005.visualizePolicy()

    # test epsilon = 0.1
    sarsa_e01 = GridWorld_SARSA(episodes=10000, alpha=0.1, epsilon=0.1, p1=p1, p2=p2)
    sarsa_e01.runEpisodes()
    sarsa_e01.visualizePolicy()

    # test epsilon = 0.2
    sarsa_e02 = GridWorld_SARSA(episodes=10000, alpha=0.1, epsilon=0.2, p1=p1, p2=p2)
    sarsa_e02.runEpisodes()
    sarsa_e02.visualizePolicy()
    
    