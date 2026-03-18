import random
import time


class GridWorld_QLearning:
    def __init__(self, episodes, alpha=0.1, gamma=0.9, p1=1.0, p2=0.0, epsilon=0.1):
        assert p1 + p2 <= 1.0

        self.grid = self.generateGrid()
        self.qTable = self.generateQTable()

        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.p1 = p1
        self.p2 = p2
        self.epsilon = epsilon

        self.agentPosition = self.initializeAgentPosition()
        self.manhattanDistance = 1

    def initializeAgentPosition(self):
        while True:
            r = random.randint(0, 10)
            c = random.randint(0, 10)

            if self.grid[r][c] == " ":
                self.agentPosition = (r, c)
                self.manhattanDistance = r + (11 - c)
                if self.manhattanDistance <= 0:
                    self.manhattanDistance = 1
                return self.agentPosition

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

    def generateQTable(self):
        qTable = {}
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] in [" ", "G"]:
                    qTable[(r, c)] = [0.0, 0.0, 0.0, 0.0]
        return qTable

    def adjacentActions(self, a):
        return [2, 3] if a in [0, 1] else [0, 1]

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            pref = random.choice([0, 1, 2, 3])
        else:
            q = self.qTable[state]
            pref = q.index(max(q))

        adj = self.adjacentActions(pref)
        adjProb = (1 - self.p1 - self.p2) / 2

        return random.choices(
            [pref, -1, adj[0], adj[1]],
            weights=[self.p1, self.p2, adjProb, adjProb],
            k=1
        )[0]

    def canUpdatePos(self, pos):
        r, c = pos
        return 0 <= r <= 10 and 0 <= c <= 10 and self.grid[r][c] in [" ", "G"]

    def calculateReward(self, pos):
        return 500 if self.grid[pos[0]][pos[1]] == "G" else -1

    def step(self, move):
        delta = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),-1:(0,0)}

        dr, dc = delta[move]
        new = (self.agentPosition[0]+dr, self.agentPosition[1]+dc)

        if self.canUpdatePos(new):
            self.agentPosition = new

        r = self.calculateReward(self.agentPosition)
        done = r == 500

        return self.agentPosition, r, done

    # Q-learning update (off-policy)
    def runEpisode(self):
        state = self.initializeAgentPosition()
        action = random.choice([0,1,2,3])
        done = False
        steps = 0

        while not done:
            next_state, reward, done = self.step(action)
            steps += 1

            if done:
                target = reward
            else:
                target = reward + self.gamma * max(self.qTable[next_state])

            self.qTable[state][action] += self.alpha * (target - self.qTable[state][action])

            state = next_state
            if not done:
                action = self.chooseAction(state)

        return steps

    def visualizePolicy(self):
        arrows = {0:"^",1:"v",2:"<",3:">"}

        print("\n--- Learned Policy ---")

        for r in range(len(self.grid)):
            row = []
            for c in range(len(self.grid[r])):
                cell = self.grid[r][c]

                if cell == "G":
                    row.append("G")
                elif cell in ["|","-","+"]:
                    row.append(cell)
                else:
                    q = self.qTable[(r,c)]
                    row.append("?" if q == [0,0,0,0] else arrows[q.index(max(q))])

            print(" ".join(row))
        print()

    def runEpisodes(self):
        print("Q-Learning running...")

        totalSteps = 0
        totalTime = 0

        for i in range(1, self.episodes+1):
            start = time.time()
            steps = self.runEpisode()
            end = time.time()

            totalSteps += steps
            totalTime += (end - start)

            if i % 500 == 0:
                print(f"Episode {i}: avg steps = {totalSteps/i:.2f}")

        print("\nFinal Stats:")
        print("Episodes:", self.episodes)
        print("Steps:", totalSteps)
        print("Time:", totalTime)


if __name__ == "__main__":
    p1 = float(input("Enter p1: "))
    p2 = float(input("Enter p2: "))

    # epsilon = 0.05
    q_e005 = GridWorld_QLearning(episodes=5000, alpha=0.1, epsilon=0.05, p1=p1, p2=p2)
    q_e005.runEpisodes()
    q_e005.visualizePolicy()

    # epsilon = 0.1
    q_e01 = GridWorld_QLearning(episodes=5000, alpha=0.1, epsilon=0.1, p1=p1, p2=p2)
    q_e01.runEpisodes()
    q_e01.visualizePolicy()

    # epsilon = 0.2
    q_e02 = GridWorld_QLearning(episodes=5000, alpha=0.1, epsilon=0.2, p1=p1, p2=p2)
    q_e02.runEpisodes()
    q_e02.visualizePolicy()