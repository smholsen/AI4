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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def max_value(other_game_state, current_depth):
            current_depth += 1

            if is_fin(other_game_state, current_depth):
                return other_game_state.getScore()
            value = float('-Inf')
            for action in other_game_state.getLegalActions(0):
                value = max(value, min_value(other_game_state.generateSuccessor(0, action), current_depth, 1))
            return value

        def min_value(other_game_state, current_depth, agent_index):
            if is_fin(other_game_state, current_depth):
                return other_game_state.getScore()
            value = float('Inf')

            for action in other_game_state.getLegalActions(agent_index):
                # We have now iterated through all the ghosts, so now pacman.
                if agent_index == number_of_ghosts:
                    value = min(value, max_value(other_game_state.generateSuccessor(agent_index, action), current_depth))
                else:
                    # Still calculating for ghost. Run again after for next agent.
                    value = min(value, min_value(other_game_state.generateSuccessor(agent_index, action), current_depth, agent_index + 1))
            return value

        # Method decides whether game is finished or not. If it is, return score for current state.
        def is_fin(other_game_state, other_depth):
            return other_game_state.isWin() or other_game_state.isLose() or other_depth == self.depth

        def minimax(start_game_state):
            return_action = None

            best_value = -float('Inf')
            for possible_action in start_game_state.getLegalActions(0):
                current_depth = 0

                value = min_value(start_game_state.generateSuccessor(0, possible_action), current_depth, 1)
                if value > best_value:
                    best_value = value
                    return_action = possible_action

            return return_action

        number_of_ghosts = gameState.getNumAgents() - 1
        return minimax(gameState)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(other_game_state, current_depth, alpha, beta):
            current_depth += 1

            if is_fin(other_game_state, current_depth):
                return self.evaluationFunction(other_game_state)
            value = float('-Inf')
            for action in other_game_state.getLegalActions(0):
                value = max(value, min_value(other_game_state.generateSuccessor(0, action), current_depth, 1, alpha,
                                             beta))
                if cutoff_test_beta(value, beta):
                    return value
                alpha = max(value, alpha)
            return value

        def min_value(other_game_state, current_depth, agent_index, alpha, beta):
            if is_fin(other_game_state, current_depth):
                return self.evaluationFunction(other_game_state)
            value = float('Inf')

            for action in other_game_state.getLegalActions(agent_index):
                # We have now iterated through all the ghosts, so now pacman.
                if agent_index == number_of_ghosts:
                    value = min(value, max_value(other_game_state.generateSuccessor(agent_index, action), current_depth,
                                                 alpha, beta))
                else:
                    # Still calculating for ghost. Run again after for next agent.
                    value = min(value, min_value(other_game_state.generateSuccessor(agent_index, action), current_depth,
                                                 agent_index + 1, alpha, beta))
                    if cutoff_test_alpha(value, alpha):
                        return value
                    beta = min(value, beta)
            return value

        # Method decides whether game is finished or not. If it is, return score for current state.
        def is_fin(other_game_state, other_depth):
            return other_game_state.isWin() or other_game_state.isLose() or other_depth == self.depth

        # Method that decides whether or not the algo should continue on current node for alpha test
        # Returns true if algo should cut.
        def cutoff_test_alpha(value, alpha):
            return value <= alpha

        # Method that decides whether or not the algo should continue on current node for beta test
        # Returns true if algo should cut.
        def cutoff_test_beta(value, beta):
            return value >= beta

        def alpha_beta_prune(start_game_state):
            return_action = None

            best_value = -float('Inf')
            alpha = float('-Inf')
            beta = float('Inf')

            for possible_action in start_game_state.getLegalActions(0):
                current_depth = 0

                value = min_value(start_game_state.generateSuccessor(0, possible_action), current_depth, 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    return_action = possible_action

            return return_action

        number_of_ghosts = gameState.getNumAgents() - 1
        return alpha_beta_prune(gameState)

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

