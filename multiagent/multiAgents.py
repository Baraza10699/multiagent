# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if currentGameState.isWin():
            return currentGameState.getScore()

        for ghostState in newGhostStates:
            if manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return - sys.maxint - 1

        newFoodList = newFood.asList()
        sum = - len(newFoodList)

        minDist = sys.maxint
        for element in newFoodList:
            manDist = manhattanDistance(element, newPos)
            if manDist < minDist:
                minDist = manDist

        sum += 0.99 / float(minDist)

        return sum

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minmax(gameState, agent, depth, a, b):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState), 0
            if depth == self.depth:
                return self.evaluationFunction(gameState), 0
            if agent == gameState.getNumAgents() - 1:
                depth += 1
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index
            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result:
                    nextValue = minmax(gameState.generateSuccessor(agent, action), nextAgent, depth, a, b)
                    result.append(nextValue[0])
                    result.append(action)
                    if agent == self.index:
                        a = max(result[0], a)
                    else:
                        b = min(result[0], b)
                else:
                    previousValue = result[0]
                    nextValue = minmax(gameState.generateSuccessor(agent, action), nextAgent, depth, a, b)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            a = max(result[0], a)
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            b = min(result[0], b)
            return result

        return minmax(gameState, self.index, 0, -float("inf"), float("inf"))[1]

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def AB(gameState, agent, depth, a, b):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState), 0
            if depth == self.depth:
                return self.evaluationFunction(gameState), 0
            if agent == gameState.getNumAgents() - 1:
                depth += 1
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index
            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result:
                    nextValue = AB(gameState.generateSuccessor(agent, action), nextAgent, depth, a, b)
                    result.append(nextValue[0])
                    result.append(action)
                    if agent == self.index:
                        a = max(result[0], a)
                    else:
                        b = min(result[0], b)
                else:

                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result
                    previousValue = result[0]
                    nextValue = AB(gameState.generateSuccessor(agent, action), nextAgent, depth, a, b)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            a = max(result[0], a)
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            b = min(result[0], b)
            return result

        return AB(gameState, self.index, 0, -float("inf"), float("inf"))[1]
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectiMax(gameState, agent, depth):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState), 0

            if depth == self.depth:
                return self.evaluationFunction(gameState), 0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index


            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result:
                    nextValue = expectiMax(gameState.generateSuccessor(agent, action), nextAgent, depth)

                    if (agent != self.index):
                        result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                        result.append(action)
                    else:

                        result.append(nextValue[0])
                        result.append(action)
                else:

                    previousValue = result[0]
                    nextValue = expectiMax(gameState.generateSuccessor(agent, action), nextAgent, depth)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action


                    else:
                        result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                        result[1] = action
            return result

        return expectiMax(gameState, self.index, 0)[1]
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <I have used the following things to my advantage:
      1) Ghosts in Pacman are pretty slow. They are only as fast as Pacman. So the only bad state in terms of ghost is the state adjacent to a ghost.
      2) If you eat a scared ghost you get 200 points, so having a scared ghost near you is good.
      3) Having a food Particle near you is good but not as good as eating them which is 10 points.
      >
    """
    "*** YOUR CODE HERE ***"

    class nodeClass:
        def __init__(self, state, pathCost):
            self.state = state
            self.pathCost = pathCost

    def breadthFirstSearchLength(startState, goalList,
                                 walls):  # Returns Length of path to the closest food Particle using BFS
        """Search the shallowest nodes in the search tree first."""
        node = nodeClass(startState, 0)
        frontier = util.Queue()
        frontier.push(node)
        explored = []
        while True:
            if frontier.isEmpty():
                return None
            node = frontier.pop()
            if node.state in goalList:
                return node.pathCost
            if node.state not in explored:
                explored.append(node.state)
                for state in [(node.state[0] - 1, node.state[1]), (node.state[0] + 1, node.state[1]),
                              (node.state[0], node.state[1] - 1),
                              (node.state[0], node.state[1] + 1)]:
                    if state not in walls:
                        child = nodeClass(state, node.pathCost + 1)
                        frontier.push(child)

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    wallList = currentGameState.getWalls().asList()
    ghostStates = currentGameState.getGhostStates()

    if currentGameState.isWin():  # Just return the score if you win
        return currentGameState.getScore()

    sum = 0
    for ghostState in ghostStates:
        if ghostState.scaredTimer == 0:
            if manhattanDistance(ghostState.getPosition(),
                                 pos) < 2:  # If an Unscared Ghost is adjacent to you, then that is a very bad state
                return - sys.maxint - 1
        else:
            dist = manhattanDistance(pos, ghostState.getPosition())
            if ghostState.scaredTimer > dist:  # If you can chase a scaredGhost before he gets un-scared
                sum += 200.0 / dist  # Then, the closer you are to the scaredGhost the Better

    sum += 9.99 / breadthFirstSearchLength(pos, foodList,
                                           wallList)  # Closer you are to food the better, but never better than actually eating food.(which has a score of 10)
    sum += currentGameState.getScore()  # The more your score the better

    return sum
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
