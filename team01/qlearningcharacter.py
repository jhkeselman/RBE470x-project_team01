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
import copy
import random

class QLearningCharacter(CharacterEntity):

    # features = ["MONSTER_DISTANCE", "EXIT_DISTANCE", "EXPLORABLE_DISTANCE_FROM_EXIT", "BOMB_DISTANCE", "WALL_DISTANCE", "EXPLOSION_DISTANCE","BOMB_PLACED"]
    
    features=["EXIT_DISTANCE", "IS_ALIVE", "NUMBER_OF_WALLS", "BOMB_PLACED", "IN_BOMB_RANGE", "DISTANCE_FROM_BOMB","DISTANCE_FROM_MONSTER"]
    num_features = len(features)
    weights = []
    
    diagactions = [[[0,1],0], [[0,-1],0], [[-1,0],0], [[1,0],0], [[0,0],1], [[1,1],0], [[-1,1],0], [[1,-1],0], [[-1,-1],0]]
    actions = [[[0,1],0], [[0,-1],0], [[-1,0],0], [[1,0],0], [[0,0],1]]#,[[0,1],1], [[0,-1],1], [[-1,0],1], [[1,0],1]]#, [[1,1],0], [[-1,1],0], [[1,-1],0], [[-1,-1],0]]# Static for now
    actions = diagactions # enable diagonal movement

    # Store values first time world is seen
    init_flag = False
    exit_wavefront = None # Wavefront from exit, ignores walls  
    to_goal_wave = None # Wavefront from exit, observes wall
    world_size = None # Number of cells in the world. Used to normalize distance values
    goal = None # Coordinates of the exit
    isBomb=False
    maxTime=0

    # Store values that change every turn
    position = None # Current position
    reachable = None # List of reachable cells from current position
    
    def __init__(self, name, color, x, y):
        super().__init__(name, color, x, y)
        try:
            self.weights = []
            self.weights = np.loadtxt("weights.csv",
                 delimiter=",", dtype=float)
            print("Inital Weighs: ",self.weights)
            if len(self.weights) != self.num_features:
                raise Exception("Weights not the same length as features")

            if np.isnan(self.weights).any():
                raise Exception("Weights contain nan")

        except:
            print("Picking random weights")
            self.weights = []
            for i in range(self.num_features):
                self.weights.append(random.random())
            np.savetxt("weights.csv",self.weights)


    def do(self, wrld):
        if not self.init_flag:
            self.goal = self.findExit(wrld)
            self.exit_wavefront = self.generateWavefront(wrld, self.goal,True)
            self.to_goal_wave = self.generateWavefront(wrld,self.goal,False)
            self.world_size = np.max(self.exit_wavefront)
            self.maxWalls = self.findWalls(wrld)
            self.init_flag = True
            self.isBomb=False
            self.maxTime=wrld.time
        

        
        if wrld.explosions.__len__()>0:
            self.exit_wavefront = self.generateWavefront(wrld, self.goal,True)
            "redone wavefronts"
            # exit()
        
        #Bomb hysteresis to update the world after bomb goes off
        # if not self.isBomb and self.findBomb(wrld):
        #     self.isBomb=True
        # if self.isBomb and not self.findBomb(wrld):
        #     self.isBomb=False
        #     self.exit_wavefront = self.generateWavefront(wrld, self.goal,True)
        #     self.to_goal_wave = self.generateWavefront(wrld,self.goal,False)
            

        self.position = (self.x, self.y)

        dx, dy = 0, 0
        bomb = False
        

        path = self.astar(wrld, self.position, self.goal, withExplosions=True)

        monsters = self.findMonsters(wrld)
        print("Path: ", path)

        

# --------------------------------------------------------------------------
        if(path != []) and (monsters == []):
            nextPoint = path.pop(1)
            dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
            self.move(dx, dy)
            return
        elif (path != []):
            dist_to_goal = len(path)
            monst_dist_to_goal = float('inf')
            for monster in monsters:
                this_mon_dist=len(self.astar(wrld, monster, self.goal))
                if this_mon_dist>0:
                    monst_dist_to_goal = min(monst_dist_to_goal,this_mon_dist)
            if dist_to_goal<monst_dist_to_goal - 2:
                nextPoint = path.pop(1)
                dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
                self.move(dx, dy)
                return
        # print("Choosing Action:")
        result = self.argMax([(wrld, action) for action in self.actions], self.getQValue)
        
        # print("Picked Action: ",action)
        if result is None: # No randomness
        # if result is None or random.random() < 0.1:
            # print("Random action")
            action = self.actions[random.randint(0,len(self.actions)-1)]
        else:
            action = result[1]
            # print("Picked Action: ",action)
        dx, dy = action[0]
        bomb = action[1]
        
        delta = self.getDelta(wrld, action)
        # print("updating weights: ")
        self.updateWeights(self.getFeatureValues(wrld), delta)
        try:
            np.savetxt("weights.csv",self.weights)
        except:
            pass
        
        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

    def getDelta(self, wrld, action_taken):
        gamma = 0.9
        # print("Args: ",[(wrld, action) for action in self.actions])
        new_wrld = self.result(wrld, action_taken)
        arg_max = self.argMax([(new_wrld, action) for action in self.actions], self.getQValue)
        if arg_max is None:

            arg_max = self.actions[random.randint(0,len(self.actions)-1)]
        else:
            arg_max = arg_max[1]

        r = self.getReward(new_wrld)#wrld.scores["me"]
        return r + gamma * self.getQValue(new_wrld, arg_max) - self.getQValue(wrld, action_taken)

    def updateWeights(self, feature_values, delta):
        alpha = 0.05
        for i in range(self.num_features):
            self.weights[i] += alpha * -delta * feature_values[i]   
    
    def getQValue(self, wrld, action):
        # print("Getting result of action: ",action)
        # print("Getting Q Value ", end = " ")
        next = self.result(wrld, action)
        # print("From action: ",action, end = " ")
        if next is None:
            # print("Returning -10000 for no world")
            return -10000
        # print("Next next: ", next.next())
        if (self.findChar(next.next()[0]) == (-1,-1)):
            # print("Returning -10000 for dead stat: character: ",self.findChar(next.next()[0]) )
            # return float('-inf')
            return -10000
        # print("Next world: ",next)
        
        # print("self.weights: ",self.weights)
        # print("q value: ",np.dot(self.weights, self.getFeatureValues(next)),end="\n")
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
        if feat_name=="EXIT_DISTANCE":
            dist = self.exit_wavefront[pos[0]][pos[1]]
            # print("Exit distance: ",dist)
            return (self.world_size-dist)/self.world_size
        if feat_name=="BOMB_PLACED":
            if self.findBomb(wrld):
                return 1.0
            return 0
        if feat_name=="IS_ALIVE":
            if self.findChar(wrld) == (-1,-1):
                return 0.0
            return 1.0
        if feat_name=="NUMBER_OF_WALLS":
            count = self.findWalls(wrld)
            return (self.maxWalls-count)/self.maxWalls
          
        if feat_name=="IN_BOMB_RANGE":
            bomb = self.findBomb(wrld)
            if bomb:
                timeToExplode = self.checkTimeToExplode(wrld, bomb, pos)
                if timeToExplode>=1:
                    return 1-1/timeToExplode
            return 1.0
        
        if feat_name=="DISTANCE_FROM_BOMB":
            bomb = self.findBomb(wrld)
            explosion = self.findExplosion(wrld)
            if explosion:
                dist = float('inf')
                for expl in explosion:
                    thisdist = len(self.astar(wrld, pos, (expl[0],expl[1])))
                    if thisdist>0:
                        dist = min(dist,thisdist)
                if dist != float('inf'):
                    return 1-1/(dist+.1)
            if bomb:
                dist=len(self.astar(wrld, pos, bomb))
                return 1-1/(dist+.1)
            
            return 1
        
        if feat_name=="DISTANCE_FROM_MONSTER":
            monsters = self.findMonsters(wrld)
            if monsters:
                dist = float('inf')
                for monster in monsters:
                    thisdist = len(self.astar(wrld, pos, monster,throughWalls=False))
                    if thisdist>0:
                        dist = min(dist,thisdist)
                return 1-1/(dist)
            return 0
        
        return 0  

    def getReward(self, wrld):
        """
        Returns the reward of the given world state.

        Args:
            wrld (World): The game world.

        Returns:
            int: The reward of the world state.
        """
        charx, chary = self.findChar(wrld)
        if charx == -1 and chary == -1:
            return -10000
        bombPoints = 0
        bomb = self.findBomb(wrld)
        
        if bomb:
            bombPoints = 500
            dist=len(self.astar(wrld, (charx,chary), bomb))
            if dist>0:
                bombPoints = 500+2*dist

            
        selfDistToGoal = self.exit_wavefront[charx][chary]
        
        if wrld.explosion_at(charx, chary):
            return -10000  
        if bomb and self.checkTimeToExplode(wrld, bomb, (charx, chary)) == 1:
            return -10000   
        monstPoints = 0#self.world_size
        monsters = self.findMonsters(wrld)
        if monsters:
            dist = float('inf')
            for monster in monsters:
                    thisdist = len(self.astar(wrld, (charx,chary), monster,throughWalls=False,withExplosions=True))
                    if thisdist>0:
                        dist = min(dist,thisdist)            
            if dist != float('inf'):
                monstPoints = 6*dist
        timepassed = wrld.time-self.maxTime   
        return wrld.scores["me"] - selfDistToGoal + bombPoints + monstPoints
        



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
        try:
            newWorld = wrld.from_world(wrld)
            nx, ny = newWorld.me(self).x, newWorld.me(self).y
            wavefront = self.generateWavefront(newWorld, (nx, ny))
            reachable = []
            for row in range(len(wavefront)):
                for col in range(len(wavefront[0])):
                    if wavefront[row][col] < float('inf'):
                        reachable.append((col, row))
            return reachable
        except Exception as e:
            print("Error in reachableCells: ",e)
            return []


    def astar(self, wrld, start, goal,throughWalls=False,withExplosions=False):

        """
        A* algorithm for finding the shortest path from start to goal in a given world.

        Args:
            wrld (World): The game world.
            start (tuple): The starting position.
            goal (tuple): The goal position.
        
        Returns:
            list: The shortest path from start to goal as a list of positions.
        """
        path = []
        found = False

        pq = PriorityQueue()
        pq.put((0,tuple(start), None), 0)
        explored = {}
        while not found and not pq.empty():
            element = pq.get()
            g = element[0]
            exploring = element[1]
            explored[exploring] = element
            if exploring == tuple(goal):
                found = True
                break


            neighbors = self.getNeighbors(wrld, exploring, throughWalls=throughWalls,withExplosions=withExplosions, turns=1)

            # print(neighbors)
            for neighbor in neighbors:
                if not neighbor in explored.keys():
                    gfactor = 1
                    if throughWalls and wrld.wall_at(neighbor[0], neighbor[1]):
                        gfactor+=wrld.bomb_time
                    f = g + gfactor + self.manhattan(neighbor, goal)
                    pq.put((g+gfactor, neighbor, exploring), f)
                # print(pq.get_queue())
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
        return path

    #Helper function to return the walkable neighbors 

    def getNeighbors(self, wrld, cell, withBomb=False, withMonster=False, throughWalls=False, withExplosions=False,turns=1):
 
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

        directions = self.actions
        if diag:
            directions = self.diagactions
        for action in directions:
            dir=action[0]
            newCell = (cellx + dir[0], celly + dir[1])
            if 0 <= newCell[0] < cols and 0 <= newCell[1] < rows:

                if not withExplosions or not wrld.explosion_at(newCell[0], newCell[1]):
                    if throughWalls or wrld.wall_at(newCell[0], newCell[1]) == 0:
                        if not withBomb or 0 > self.checkTimeToExplode(wrld, self.findBomb(wrld), newCell, turns):
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
                cords = element[2]
                if cords == None:
                    # This should never happen given the way the algorithm is implemented
                    return []
            path = [list(start)] + path
            # print("Path after reconstruction: ",path)
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
        
        actions = self.actions
        while not queue.empty():
            element = queue.get()
            cost = element[0]
            curCol, curRow = element[1]
            wavefront[curCol][curRow] = cost
            for action in actions:
                direction = action[0]
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
    
    def findExplosion(self, wrld):
        """
        Finds and returns the positions of all explosions in the given world.

        Parameters:
        - wrld (World): The game world.

        Returns:
        - list: A list of tuples representing the positions of the explosions.
        """
        explosions = []
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.explosion_at(col, row):
                    explosions.append((col, row))
        return explosions

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

    def checkTimeToExplode(self, wrld, bomb, tile):

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
        if wrld.explosion_at(tile[0], tile[1]):
            return 1
        return -1


    def findWalls(self,wrld):
        count = 0
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.wall_at(col, row):
                    count += 1
        return count

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
            # print("Value change: ",self.getFeatureValues(newWorld),self.getFeatureValues(nextWorld))
            # print("Posistion change: ",nextWorld.me(self).x - newWorld.me(self).x, nextWorld.me(self).y - newWorld.me(self).y)
            return nextWorld
        except Exception as e:
            return None
            print("Layer 1: ",e)
            print("Action: ",action)
            print("World: ====================================================",newWorld)
            # newWorld.printit()
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
        # print(args)
        for arg in args:
            val = util_function(*arg)
            # print(val)
            if(val > max_val):
                max_val = val
                arg_max = arg
        return arg_max