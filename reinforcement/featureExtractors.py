# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
          Dictionary includes a single feature that
          is the state,action pair. This feature doesn't
          permit generalization.
        """
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
          Dictionary includes a feature for this
          exact state, this exact action, as well
          as the x coordinate and y coordinate of
          the state.
        """
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here it's all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Returns simple features for a basic reflex Pacman:
        - whether food will be eaten
        - how far away the next food is
        - whether a ghost collision is imminent
        - whether a ghost is one step away
        """
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class AdvancedFeatureExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Returns simple features for a basic reflex Pacman:
        - whether food will be eaten
        - how far away the next food is
        - whether a ghost is one step away
        - whether the ghost is scared
        - how far a capsule is

        The weights (from a learning set of 50) are
        {'closest-food': -7.324690163475542, // The closer the closest food the better
        'ghost-scared': -31.657843653059256, // ScaredTime usually decreases as time goes on and correlate with larger gain.
                                            // Therefore we obtain the negative weight. But this variable is mainly to control
                                            // eats-food and closest-food.
        'bias': 205.92219167583784,
        '#-of-ghosts-1-step-away': -169.8062078133534, // The fewer ghosts step-away the better
        'closest-capsule': -2.040094126821756, // the closer the closest capsule is, the better
        'eats-food': 59.134306789728555} // we should eat food essentially
        """

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        '''
        compute how scared the ghost is, with the magnitude of current min scaredTime for the ghosts
        divided by the initial scare Time (this distinction is necessary because it is
        possible that pacman eats a ghost whose scare time will be reset to 0)
        '''

        from pacman import SCARED_TIME
        scaredTimers = [ghost.scaredTimer for ghost in state.getGhostStates()]
        features["ghost-scared"] = float(min(scaredTimers))/SCARED_TIME
        if features["ghost-scared"] > 0.05:
            features["eats-food"] = 5.0
            if dist is not None:
                features["closest-food"] = 5 * float(dist) / (walls.width * walls.height)
            features["#-of-ghosts-1-step-away"] = 0.

        '''
        compute the distance of the closest capsule, and we care about it because we now have the
        feature of ghost-scared, and we would prefer eating a capsule if possible.
        The magnitude is the actual min distance to the closest capsule (and will be normalized later)
        '''
        capsules = state.getCapsules()
        # We looked up the implementation of Grid object in game.py to decide how the coordinates are encoded
        capsulesMatrix = [[(x, y) in capsules for y in xrange(walls.height)] for x in xrange(walls.width)]
        dist = closestFood((next_x, next_y), capsulesMatrix, walls)
        if dist is not None and features["eats-food"]:
            features["closest-capsule"] = float(dist) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features