# valueIterationAgents.py
# -----------------------
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


import mdp, util
from util import PriorityQueue, Counter

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        """
        Value iteration implementation (batch version). First copy the old
        values and then update.
        """
        "*** YOUR CODE HERE ***"
        statelist = self.mdp.getStates()
        for state in statelist:
            self.values[state] = 0
        for k in range(iterations):
            oldValues = Counter(self.values)
            for state in statelist:
                possibleActions = self.mdp.getPossibleActions(state)
                # use a priority queue to find the max of reward
                actionRewards = PriorityQueue()
                if possibleActions == ():
                    continue
                for action in possibleActions:
                    reward = 0
                    statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState, prob in statesProbs:
                        reward += prob*(self.mdp.getReward(state,action,nextState)
                                        +self.discount*oldValues[nextState])
                    actionRewards.push(reward, -reward)
                maxRewards = actionRewards.pop()
                self.values[state] = maxRewards

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        reward = 0
        # Plug the values in the equation to get the value
        for nextState, prob in statesProbs:
            reward += prob * (self.mdp.getReward(state, action, nextState)
                              + self.discount * self.values[nextState])
        return reward

    def computeActionFromValues(self, state):
        """
          Compute the best action at a state.

          Return None if there are no possible actions. Tie break
          by first occurences.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        qRewards = PriorityQueue()
        if possibleActions == ():
            return None
        for action in possibleActions:
            reward = self.computeQValueFromValues(state, action)
            qRewards.push(action, -reward)
        optimalAction = qRewards.pop()
        return optimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
