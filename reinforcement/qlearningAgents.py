# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # A dict of util.Counter to store q values mapping state to
        # Counter's mapping actions to qValue
        self.qValues = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action).
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise.
          If a state has not been seen, add it to the dValues dict
          and initialize all legal actions to 0.
          Raise exception if the action is not legal at the state.
        """
        "*** YOUR CODE HERE ***"
        # If a state has not been seen yet
        if state not in self.qValues:
            self.qValues[state] = util.Counter()
            legalActions = self.getLegalActions(state)
            if (not legalActions):
                # terminal state
                return 0.0
            # initialize all legal actions
            if action not in legalActions:
                raise Exception("Illegal Action")
            for la in legalActions:
                self.qValues[state][la] = 0.0
        return self.qValues[state][action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action).
          where the max is over legal actions.
          If there're no legal actions, return 0.0.
        """
        "*** YOUR CODE HERE ***"
        maxAction = self.computeActionFromQValues(state)
        # print maxAction
        if maxAction is None:
            return 0.0
        else:
            return self.getQValue(state, maxAction)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
          By definition, unvisited actions has qValue of 0. and is seen as
          more optimal than negatively valued actions.
          Break tie by randomly choosing from best actions.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Find the best action
        maxAction = None
        for action in legalActions:
            if maxAction is None or self.getQValue(state, action)>self.getQValue(state, maxAction):
                maxAction = action
        maxVal = self.getQValue(state, maxAction)
        maxActions = []
        # check if there are multiple good ones
        for action in legalActions:
            if self.getQValue(state, action)==maxVal:
                maxActions.append(action)
        if len(maxActions)>1:
            return random.choice(maxActions)
        else:
            return maxAction

    def getAction(self, state):
        """
          epsilon-greedy policy to getAction.
          If there are no legal actions, return None.
          With epsilon probability we pick a random action;
          with 1-epsilon probability we choose the optimal action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        action = None
        "*** YOUR CODE HERE ***"
        randomPolicy = util.flipCoin(self.epsilon)
        if not randomPolicy:
            action = self.computeActionFromQValues(state)
        if randomPolicy:
            action = random.choice(legalActions)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          qValues is updated according to the QLearning formula.
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.computeValueFromQValues(nextState)

        if state not in self.qValues:
            self.qValues[state] = util.Counter()
        if action not in self.qValues[state]:
            self.qValues[state][action] = 0.0
        self.qValues[state][action] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    
    """ For Q7: Change the default values of epsilon and alpha in the signature below
    so that on smallGrid, the qlearning agent wins at least 80% of the time. 
    You can change gamma if you wish, but you don't need to. The "YOUR CODE HERE" is
    to mark the function, but you don't actually need to write new code- just change 
    the values.
    """
    "*** YOUR CODE HERE ***"
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Return Q(state,action) = w * featureVector.
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        return features * self.weights

    def update(self, state, action, nextState, reward):
        """
           Update the weights (in batch) base on the transition observed.
        """
        "*** YOUR CODE HERE ***"
        oldWeights = self.weights.copy()
        nextStateValue = self.getValue(nextState)
        difference = (reward  + self.discount * nextStateValue) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] = oldWeights[feature] + self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print self.weights
