# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import numpy as np
from priority_queue import PriorityQueue
from enum import Enum
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

class QLearningCharacter(CharacterEntity):

    features = ["MONSTER_DISTANCE", "EXIT_DISTANCE", "EXPLORABLE_DISTANCE_FROM_EXIT", "BOMB_DISTANCE", "WALL_DISTANCE", "EXPLOSION_DISTANCE","BOMB_PLACED"]
    num_features = len(features)
    weights = []
    

    actions = [[[0,1],0], [[0,-1],0], [[-1,0],0], [[1,0],0], [[0,0],1]] # Static for now

    # Store values first time world is seen
    init_flag = False
    exit_wavefront = None # Wavefront from exit, ignores walls  
    world_size = None # Number of cells in the world. Used to normalize distance values
    goal = None # Coordinates of the exit
    flipwave = None # Transpose of wavefront
    prevdist=float('inf')

    # Store values that change every turn
    position = None # Current position
    reachable = None # List of reachable cells from current position
    
    def __init__(self, name, color, x, y):
        super().__init__(name, color, x, y)
        try:
            self.weights = np.load("weights.npy")
        except:
            for i in range(self.num_features):
                self.weights.append(random.random())
        

    def do(self, wrld):
        if not self.init_flag:
            self.world_size = wrld.width() * wrld.height()
            self.goal = self.findExit(wrld)
            self.exit_wavefront = self.generateWavefront(wrld, self.goal,True)
            self.flipwave=np.transpose(self.exit_wavefront)
            self.init_flag = True
            self.prevdist=self.world_size
                

        # Commands
        
        # goal=self.findExit(wrld)
        # wavefront = self.generateWavefront(wrld, self.goal,True)
        self.position = (self.y, self.x)
        self.reachable = self.reachableCells(wrld, self.position)

        # print("Feature values: ",self.getFeatureValues(wrld))
        print("Weights: ",self.weights)

        dx, dy = 0, 0
        bomb = False
        # Handle input
        # for c in input("How would you like to move (w=up,a=left,s=down,d=right,b=bomb)? "):
        #     if 'w' == c:
        #         dy -= 1
        #     if 'a' == c:
        #         dx -= 1
        #     if 's' == c:
        #         dy += 1
        #     if 'd' == c:
        #         dx += 1
        #     if 'b' == c:
        #         bomb = True

        action = self.argMax([(wrld, action) for action in self.actions], self.getQValue)
        # print("=================================================================")
        print("Picked Action: ",action)
        if action is None or random.random() < 0.3:
            print("Random action")
            action = self.actions[random.randint(0,4)]
        else:
            action = action[1]
        dx, dy = action[0]
        bomb = action[1]
        
        delta = self.getDelta(wrld, action)
        self.updateWeights(self.getFeatureValues(wrld), delta)
        np.save("weights.npy",self.weights)

        # print("Action: ",action)
        

        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

    def getDelta(self, wrld, action_taken):
        gamma = 0.9
        # print("Args: ",[(wrld, action) for action in self.actions])
        
        arg_max = self.argMax([(wrld, action) for action in self.actions], self.getQValue)
        if arg_max is None:
            arg_max = self.actions[random.randint(0,4)]
        else:
            arg_max = arg_max[1]

        r = wrld.scores["me"]
        return r + gamma * self.getQValue(wrld, arg_max) - self.getQValue(wrld, action_taken)

    def updateWeights(self, feature_values, delta):
        alpha = 0.1
        for i in range(self.num_features):
            self.weights[i] += alpha * -delta * feature_values[i]
    
    def getQValue(self, wrld, action):
        # print("Getting result of action: ",action)
        next = self.result(wrld, action)
        if next is None:
            return 0
        # print("Next world: ",next)
        # print("From action: ",action)
        print("Next value: ",np.dot(self.weights, self.getFeatureValues(next)))
        return np.dot(self.weights, self.getFeatureValues(next))

    def getFeatureValues(self, wrld):
        """
        Returns the feature values of the given world state.

        Args:
            wrld (World): The game world.
            point (tuple): The coordinates of the point.
            wavefront (list): The wavefront grid.

        Returns:
            list: The features of the point.
        """
        features = []
        for i in range(self.num_features):
            features.append(self.featureValue(wrld, self.features[i]))
        return features

    def featureValue(self,wrld,feat_name):
        pos = self.findChar(wrld) # x,y

        # print("Feature value in world: ",wrld)
        if feat_name=="MONSTER_DISTANCE":
            closest = float('inf')
            for monster in self.findMonsters(wrld):
                val = len(self.astar(wrld, self.position, monster, True, False))
                if val < closest:
                    closest = val
            return np.interp(closest,[0,self.world_size],[0,1])
        if feat_name=="EXIT_DISTANCE":
            dist = self.exit_wavefront[pos[0]][pos[1]]
            
            # print("Exit distance: ",dist)
            return np.interp(dist,[0,self.world_size],[0,1])
        if feat_name=="EXPLORABLE_DISTANCE_FROM_EXIT":
            best = float('inf')
            for point in self.reachableCells(wrld, pos):  # TODO make sure uses new world
                val = self.exit_wavefront[point[0]][point[1]]
                if val < best:
                    best = val
            return np.interp(best,[0,self.world_size],[0,1])
        if feat_name=="BOMB_DISTANCE":
            dist = self.astar(wrld, pos, self.findBomb(wrld))
            if dist == []:
                return 1.0
            return np.interp(dist,[0,self.world_size],[0,1])
        if feat_name=="WALL_DISTANCE":
            return 0
        if feat_name=="EXPLOSION_DISTANCE":
            explodeTime = self.checkExplode(wrld, self.findBomb(wrld), pos)
            if explodeTime>=0:
                return 1/explodeTime
            return 0
        if feat_name=="BOMB_PLACED":
            if self.findBomb(wrld):
                return 1.0
            return 0
        return 0    

    # Copilot code. Make sure to test
    def reachableCells(self, wrld, point):
        """
        Returns the number of reachable cells from the given point.

        Args:
            wrld (World): The game world.
            point (tuple): The coordinates of the point.

        Returns:
            int: The number of reachable cells from the given point.
        """
        wavefront = self.generateWavefront(wrld, (point[0],point[1]))
        reachable = []
        for row in range(len(wavefront)):
            for col in range(len(wavefront[0])):
                if wavefront[row][col] < float('inf'):
                    reachable.append((col, row))
        return reachable

    def astar(self, wrld, start, goal, withMonster=False, withBomb=False):
        """
        A* algorithm for finding the shortest path from start to goal in a given world.

        Args:
            wrld (World): The game world.
            start (tuple): The starting position.
            goal (tuple): The goal position.
            withMonster (bool, optional): Whether to consider monsters in the pathfinding. Defaults to True.

        Returns:
            list: The shortest path from start to goal as a list of positions.
        """
        path = []
        found = False

        pq = PriorityQueue()
        pq.put((tuple(start), None, 0), 0)
        explored = {}
        if withBomb:
            bombs = self.findBomb(wrld)
        if withMonster:
            monsters = self.findMonsters(wrld)
        while not found and not pq.empty():
            element = pq.get()
            exploring = element[0]
            g = element[2]
            explored[exploring] = element
            if exploring == tuple(self.goal):
                found = True
                break

            neighbors = self.getNeighbors(wrld, exploring)
            for neighbor in neighbors:
                if explored.get(neighbor) is None or explored.get(neighbor)[2] > g + 1:
                    penalty = 0
                    if withBomb:
                        if neighbor in bombs:
                            penalty += 1000
                    if withMonster:
                        for monster in monsters:
                            dist = self.manhattan((neighbor[0], neighbor[1]), monster)
                            if dist <= 3:
                                penalty += 2 * (4 - dist)
                    f = g + 1 + self.manhattan(neighbor, self.goal) + penalty
                    pq.put((neighbor, exploring, g + 1), f)
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(self.goal))
        return path
    
    #Helper function to return the walkable neighbors 
    def getNeighbors(self, wrld, cell, withBomb=False, withMonster=False, withWalls=False, turns=1):
        """
        Returns a list of neighboring cells that are accessible from the given cell.

        Parameters:
            wrld (World): The game world.
            cell (tuple): The coordinates of the cell.

        Returns:
            list: A list of neighboring cells that are accessible from the given cell.
        """
        cellx = cell[0]
        celly = cell[1]
        neighbors = []
        rows, cols = wrld.height(), wrld.width()

        if cellx < 0 or cellx >= cols or celly < 0 or celly >= rows:
            return neighbors    

        if wrld.wall_at(cellx, celly) == 1:
            return neighbors

        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1),(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dir in directions:
            newCell = (cellx + dir[0], celly + dir[1])
            if 0 <= newCell[0] < cols and 0 <= newCell[1] < rows:
                if not withWalls or wrld.wall_at(newCell[0], newCell[1]) == 0:
                    if not withBomb or 0 > self.checkExplode(wrld, self.findBomb(wrld), newCell, turns):
                        if not withMonster or not wrld.monsters_at(newCell[0], newCell[1]):
                            neighbors.append(newCell)
        return neighbors
                            
    # Helper function to reconstruct the path from the explored dictionary
    def reconstructPath(self, explored: dict, start: tuple[int, int], goal: tuple[int, int]) -> list[list[int, int]]:   
            """
            A helper function to reconstruct the path from the explored dictionary
            :param explored [dict] The dictionary of explored nodes
            :param start [tuple(int, int)] The starting point
            :param goal [tuple(int, int)] The goal point
            :return        [list[tuple(int, int)]] The Optimal Path from start to goal.
            """
            cords = goal
            path = []
            # Loops backwards through the explored dictionary to reconstruct the path
            while cords != start:
                element = explored[cords]
                path = [list(cords)] + path
                cords = element[1]
                if cords == None:
                    # This should never happen given the way the algorithm is implemented
                    return []
            path = [list(start)] + path
            return path
    
    def generateWavefront(self, wrld, start ,throughWalls=False):
        """
        Generates a wavefront grid representing the shortest path from the start position to each cell in the world.

        Args:
            wrld (World): The game world.
            start (tuple): The starting position (row, column).

        Returns:
            list: A 2D grid representing the wavefront, where each cell contains the cost of reaching that cell from the start position.
        """
        rows, cols = wrld.height(), wrld.width()
        wavefront = [[float('inf') for _ in range(rows)] for _ in range(cols)]
        wavefront[start[0]][start[1]] = 0
        queue = PriorityQueue()
        queue.put((0, (start[0], start[1])), 0)
        
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1),(0, 1), (1, 0), (0, -1), (-1, 0)]
        while not queue.empty():
            element = queue.get()
            cost = element[0]
            curCol, curRow = element[1]
            wavefront[curCol][curRow] = cost
            for direction in directions:
                newCol, newRow = curCol + direction[0], curRow + direction[1]
                if (0 <= newRow < rows) and (0 <= newCol < cols):# and (not wrld.wall_at(newCol, newRow)):
                    if wrld.wall_at(newCol, newRow)==0:
                        newCost = cost + 1
                        if newCost < wavefront[newCol][newRow]:
                            # wavefront[newCol][newRow] = newCost
                            queue.put((newCost, (newCol, newRow)), newCost)
                    elif throughWalls:
                      
                        newCost = cost + 1 + wrld.bomb_time
                        if newCost < wavefront[newCol][newRow]:
                            # wavefront[newCol][newRow] = newCost
                            queue.put((newCost, (newCol, newRow)), newCost)
        return wavefront

    def manhattan(self, p1, p2):
        """
        Calculate the manhattan value between two points.

        Args:
            p1 (tuple): The coordinates of the first point.
            p2 (tuple): The coordinates of the second point.

        Returns:
            int: The manhattan value between the two points.
        """
        return sum([abs(p2[0] - p1[0]), abs(p2[1] - p1[1])])
    
    def openDist(self, p1, p2):
        """
        Calculates the open distance between two points.

        Parameters:
        p1 (tuple): The coordinates of the first point (x1, y1).
        p2 (tuple): The coordinates of the second point (x2, y2).

        Returns:
        int: The maximum absolute difference between the x-coordinates and the y-coordinates of the two points.
        """
        return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
        
    def findExit(self, wrld):
        """
        Finds the coordinates of the exit in the given world.

        Parameters:
        - wrld: The world object representing the game world.

        Returns:
        - exitX: The x-coordinate of the exit.
        - exitY: The y-coordinate of the exit.
        """
        exitY, exitX = 0, 0
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.exit_at(col, row):
                    exitY, exitX = row, col
        return exitX, exitY
    
    def findChar(self, wrld):
        """
        Finds the character in the given world.

        Parameters:
        wrld (World): The world object.

        Returns:
        tuple: The x and y coordinates of the character.
        """
        y, x = -1, -1
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.characters_at(col, row):
                    y, x = row, col
        return x, y
    
    def findMonsters(self, wrld):
            """
            Finds and returns the positions of all monsters in the given world.

            Parameters:
            - wrld (World): The game world.

            Returns:
            - list: A list of tuples representing the positions of the monsters.
            """
            monsters = []
            for row in range(wrld.height()):
                for col in range(wrld.width()):
                    if wrld.monsters_at(col, row):
                        monsters.append((col, row))
            return monsters
    
    def findBomb(self, wrld):
        """
        Finds the coordinates of the first bomb in the given world.

        Parameters:
            wrld (World): The game world.

        Returns:
            tuple: The coordinates (col, row) of the first bomb found, or None if no bomb is found.
        """
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.bomb_at(col, row):
                    return (col, row)
        return None

    def checkExplode(self, wrld, bomb, tile):

        """
        Checks if a bomb explosion will reach a given tile.

        Args:
            wrld (World): The game world.
            bomb (tuple): The coordinates of the bomb.
            tile (tuple): The coordinates of the tile to check.

        Returns:
            bool: True if the bomb explosion will reach the tile, False otherwise.
        """
        if bomb is None:
            return -1
        if wrld.bomb_at(bomb[0], bomb[1]).timer:
            if (abs(tile[0] - bomb[0]) <= 4 and tile[1] == bomb[1]) or (abs(tile[1] - bomb[1]) <= 4 and tile[0] == bomb[0]):
                return wrld.bomb_at(bomb[0], bomb[1]).timer

    def result(self, wrld, action):
        """
        Applies the given action to the world state and returns the resulting world state.

        Args:
            wrld (World): The current world state.
            action (tuple): The action to be applied, consisting of the movement coordinates (x, y)
                            and a flag indicating whether to place a bomb (1) or not (0).

        Returns:
            World: The resulting world state after applying the action.
            None: If an exception occurs during the action application.
        """
        newWorld = wrld.from_world(wrld)
        try:
            x, y = action[0]
            bomb = action[1]
            if bomb == 1:
                newWorld.me(self).place_bomb()
                # print("Placed bomb")
            newWorld.me(self).move(x, y)
            # print("Moved")
            nextWorld, _ = newWorld.next()
            # print("Score change: ",nextWorld.scores["me"] - newWorld.scores["me"])
            print("Value change: ",self.getFeatureValues(newWorld),self.getFeatureValues(nextWorld))
            # print("Posistion change: ",nextWorld.me(self).x - newWorld.me(self).x, nextWorld.me(self).y - newWorld.me(self).y)
            return nextWorld
        except Exception as e:
            # print("Layer 1: ",e)
            # print("Action: ",action)
            # print("World: ====================================================",newWorld)
            newWorld.printit()
            try:
                nextWorld,_ = newWorld.next()
                return nextWorld
            except Exception as e:
                # print("Layer 2: ",e)
                # print("World: ",newWorld)
                return None
    
    @staticmethod     
    def argMax(args,util_function):
        max_val = float('-inf')
        arg_max = None # Was None
        print(args)
        for arg in args:
            val = util_function(*arg)
            print(val)
            if(val > max_val):
                max_val = val
                arg_max = arg
        return arg_max