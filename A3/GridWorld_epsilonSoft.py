import random
import time


class GridWorld_epsilonSoft:
    def __init__(self, episodes, gamma=0.9, p1=None, p2=None, epsilon=0.1, epsilonDecay = 0.0):
        self.grid = self.generateGrid()
        self.nTable, self.qTable = self.generateTables()
        self.epsilon = epsilon
        self.episodes = episodes
        self.gamma = gamma
        self.agentPosition = self.initializeAgentPosition()
        self.manhattanDistance
        self.epsilonDecay = epsilonDecay

        # if the user inputs a P1 and P2, sum must be less than or equal to 1
        if p1 is not None and p2 is not None:
            assert p1 + p2 <= 1

        # P1 and P2 are none/no user input, per assignment guideline
        else:
            self.p1 = 9.0
            self.p2 = 0.1

    # start the agent off in some random state (row,col)
    def initializeAgentPosition(self): 
        while True:
            row = random.randint(0,10)
            col = random.randint(0,10)

            # agent should not spawn in a wall or terminal state G
            if self.grid[row][col] == " ":
                self.agentPosition = (row,col)
                self.manhattanDistance = (11 - row) + (11 - col)
                return self.agentPosition


    # makes the 5x5 grid, with terminal state G
    def generateGrid(self) -> None:
        # the grid will be a 10x10, but to accomodate the walls between the rooms, it will be 11x11 in the implementation
        grid = []

        # -, + and | are walls
        for _ in range(11):
            if _ == 0:
                grid.append([" ", " ", " ", " ", " ", "|", " ", " ", " ", " ", "G"])
            elif _ == 5:
                grid.append(["-","-"," ","-","-","+","-","-"," ","-", "-"])
            elif _ == 2 or _ == 8:
                grid.append([" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "])
            else:
                grid.append([" ", " ", " ", " ", " ", "|", " ", " ", " ", " ", " "])
        return grid

    def printGrid(self) -> None:
        for row in self.grid:
            print(row)
    
    # initializes Q Table
    def generateTables(self):
        qTable = {}
        nTable = {}
        gridSize = len(self.grid)

        for row in range(gridSize):
            for col in range(gridSize):
                if self.grid[row][col] == " " or self.grid[row][col] == "G":
                    # Initialize Q-values to 0.0 and N-counts to 0
                    qTable[(row, col)] = [0.0, 0.0, 0.0, 0.0]
                    nTable[(row, col)] = [0, 0, 0, 0]

        return qTable, nTable

    # if the robot slips and takes a different direction, these are the available options based on its perferred choice
    def adjacentActions(self, preferredAction):
        match preferredAction:
            case 0 | 1: # up or down
                return [2, 3]
            case 2 | 3: # left or right
                return [0, 1]

    # decides what action the agent will take
    def decideAction(self, state: tuple) -> None:
        # agent will have epsilon chance to take a random action
        exploitOrExplore = random.choices(["exploit", "explore"], weights=[1 - self.epsilon, self.epsilon], k=1)

        if exploitOrExplore[0] == "exploit":
            # if exploiting, we are taking greedy/argmax of possible directions from q table
            possibleMoves = self.qTable[state] # directions in form [v_up, v_down, v_left, v_right]
            maxEstimatedReturn = max(possibleMoves) # argmax
            preferredMove = possibleMoves.index(maxEstimatedReturn) # first occurrence if multiple argmax
        else:
            # just pick a random direction
            preferredMove = random.choice([x for x in range(4)])
    
        # because of environment, agent may not actually be able to perform the action they want, based on p1 and p2
        adjacentState = self.adjacentActions(preferredMove) # possible other moves
        adjacentProb = (1 - self.p1 - self.p2)/2
        actualMove = random.choices(
            [preferredMove, -1, adjacentState[0], adjacentState[1]],
            weights=[self.p1, self.p2, adjacentProb, adjacentProb],
            k=1
            ) # -1 for no move/stay in current state
        return actualMove[0]
    
    # make sure new position is actually in the board
    def inBounds(self, agentPos):
        newRow, newCol = agentPos
        return (newRow <= 10 and newRow >= 0) and (newCol <= 10 and newCol >= 0)

    # don't update agent position upon running into wall
    def notWall(self, agentPos):
        newRow, newCol = agentPos
        return (self.grid[newRow][newCol] == " ") or (self.grid[newRow][newCol] == "G")
    
    def canUpdatePos(self, agentPos):
        return self.inBounds(agentPos) and self.notWall(agentPos)
    
    def calculateReward(self, agentPos):
        if self.grid[agentPos[0]][agentPos[1]] == "G":
            return 500
        else:
            return -1
        
    def step(self, move):
        # to convert move (as number/index) into actual positional change
        delta = {
            0: (-1, 0),  # up
            1: (1, 0), # down
            2: (0, -1), # left
            3: (0, 1),  # right
            -1:(0, 0)   # no change
        }

        # calculate new pos
        changeHorizontal, changeVertical = delta[move]
        newPosition = (self.agentPosition[0] + changeHorizontal, self.agentPosition[1] + changeVertical)

        if self.canUpdatePos(newPosition):
            self.agentPosition = newPosition
        
        reward = self.calculateReward(self.agentPosition)
        done = reward == 500
        return (self.agentPosition, reward, done)

    # assuming first-visit MC since not stated in assignment outline
    def updateQTable(self, trajectory):
        # pre-calculate the first occurrence of each (S, A) pair using a dictionary: {(state, action): index}, faster lookup times
        first_occurrences = {}
        for idx, (state, action, reward) in enumerate(trajectory):
            if (state, action) not in first_occurrences:
                first_occurrences[(state, action)] = idx

        # Q Table update starting from terminal state
        G = 0
        for i in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[i]
            G = reward + self.gamma * G
            
            # only update if the current index is the first occurrence
            if i == first_occurrences.get((state, action)):
                self.nTable[state][action] += 1
                n = self.nTable[state][action]
                
                old_q = self.qTable[state][action]
                # Incremental Average formula
                self.qTable[state][action] = old_q + (1/n) * (G - old_q)


    
    def runEpisode(self):
        # agent position already initialized, grid initialized
        trajectory = []
        done = False
        state = self.initializeAgentPosition()

        # first move has to be random by assignment guidelines
        action = random.choice([0,1,2,3])

        while not done:
            next_state, reward, done = self.step(action)
            trajectory.append((state, action, reward))

            # prevent updates if agent has reached terminal state earlier in loop
            if not done:
                state = next_state
                action = self.decideAction(state)

        # return all of the actions
        return trajectory

    def visualizePolicy(self):
        arrow_symbols = {
            0: "^",  # 0 is UP
            1: "v",  # 1 is DOWN
            2: "<",  # 2 is LEFT
            3: ">"   # 3 is RIGHT
        }

        print("\n--- Learned Policy ---")
        grid_size = len(self.grid)

        for r in range(grid_size):
            row_output = []
            for c in range(grid_size):
                cell_char = self.grid[r][c]

                if cell_char == "G":
                    row_output.append("G") # terminal state
                elif cell_char in ["|", "-", "+"]:
                    row_output.append(cell_char) # walls
                else:
                    # It's a walkable space (" "), find the best action based on QTable
                    state = (r, c)
                    q_values = self.qTable[state]
                    
                    # If all values are 0, the agent never visited here
                    if max(q_values) == 0.0:
                        row_output.append("?") 
                    else:
                        # Find index of the max value
                        best_action = q_values.index(max(q_values))
                        row_output.append(arrow_symbols[best_action])
            
            # print row
            print(" ".join(row_output))

        # new line between "models" results
        print()
    
    def runEpisodes(self):
        print(f"MC- Epsilon Soft method (gamma = {self.gamma}, epsilon = {self.epsilon}, epsilonDecay = {self.epsilonDecay})")

        stepCount = 0
        computationalTime = 0

        for i in range(1, self.episodes + 1):
            startTime = time.time()

            trajectory = self.runEpisode()
            self.updateQTable(trajectory)

            endTime = time.time()
            episodeTime = endTime - startTime

            # statistics
            stepCount += len(trajectory)/self.manhattanDistance
            computationalTime += episodeTime/self.manhattanDistance
            
            
            # print progress every 500 episodes
            if i % 500 == 0:
                print(f'Episode: {i}: \n Average Time steps: {stepCount/i:.2f} \n Average Comp Time: {computationalTime/i:.6f}s \n Current Epsilon: {self.epsilon}')

if __name__ == "__main__":
    episodeCount = 5000
    grid_e01 = GridWorld_epsilonSoft(episodes=episodeCount, epsilon=0.1)
    grid_e01.runEpisodes()
    grid_e01.visualizePolicy()

    grid_e005 = GridWorld_epsilonSoft(episodes=episodeCount,epsilon=0.05)
    grid_e005.runEpisodes()
    grid_e005.visualizePolicy()

    grid_e02 = GridWorld_epsilonSoft(episodes=episodeCount, epsilon=0.2)
    grid_e02.runEpisodes()
    grid_e02.visualizePolicy()


