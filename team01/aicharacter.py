import sys
import math
import numpy as np
from enum import Enum
from heapq import heappush, heappop
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from priority_queue import PriorityQueue

class AICharacter(CharacterEntity):

    #  Init class variables
    optimalAction = ()
    depthMax = 3
    wave=None
    moves=[]
    wasBomb=False
    goForwards=False
    start=None

    # Take an action
    def do(self, wrld):
        
        # Store exit, find wavefront, and find path
        exitX, exitY = self.findExit(wrld)
        if self.wave is None:
            self.start=(self.x,self.y)
            self.wave=self.generateWavefront(wrld, [exitX, exitY])
        path = self.astar(wrld, [self.x, self.y], [exitX, exitY])
        
        skipAB=False

        if path:
            monsters=self.findMonsters(wrld)
            # If closer to goal than monsters, go directly to the goal
            if monsters!=[]:
                selfDistToGoal=self.wave[self.x][self.y]
                monsterDistToGoal=min([self.wave[monster[0]][monster[1]] for monster in monsters])
                if selfDistToGoal < monsterDistToGoal-2:
                    nextPoint = path.pop(1)
                    dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
                    self.move(dx, dy)
                    return
            if not self.findBomb(wrld):
                # Bomb if monster is within 5 steps
                if monsters!=[]:
                    closestMonster=(-1,-1)
                    for monster in monsters:
                        if closestMonster==(-1,-1) or self.openDist((self.x,self.y),monster)<self.openDist((self.x,self.y),closestMonster):
                            closestMonster=monster
                    if len(self.astar(wrld,self.findChar(wrld),closestMonster,False))<=5 and self.y>0 and self.x>0:
                        self.place_bomb()
                        self.wasBomb=True
                        self.goForwards=False
                        skipAB=True
                        print("bomb","skiping AB")

            # Get best move
            if (not self.findBomb(wrld) )and not skipAB:
                print("the reg")
                monsters=self.findMonsters(wrld)
                
                if not monsters: # a* to goal
                    nextPoint = path.pop(1)
                    dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
                    self.move(dx, dy)
                    return
                else:
                    selfDistToGoal=self.wave[self.x][self.y]
                    monsterDistToGoal=min([self.wave[monster[0]][monster[1]] for monster in monsters])
                    # Go fast to goal if closer than monster
                    if selfDistToGoal < monsterDistToGoal:
                        nextPoint = path.pop(1)
                        dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
                        self.move(dx, dy)
                        return
                if self.wasBomb:
                    self.wasBomb=False
                    self.wave=self.generateWavefront(wrld, [exitX, exitY])
                move, bomb = self.abSearch(wrld, self.depthMax, float('-inf'), float('inf'))
                # Move
                dx, dy = move
                self.move(dx, dy)
                self.moves.append((dx,dy))

                # Place bomb
                if bomb == 1:
                    self.place_bomb()
                    self.wasBomb=True
                    self.goForwards=False

                
            else: # if theres a bomb

                #go for the exit if we are closer to the goal than the closer monster else go back
                monsters=self.findMonsters(wrld)
                if monsters:
                    closestMonster=(-1,-1)
                    for monster in monsters:
                        if closestMonster==(-1,-1) or self.openDist((self.x,self.y),monster)<self.openDist((self.x,self.y),closestMonster):
                            closestMonster=monster
                    monsterDistToGoal=self.wave[closestMonster[0]][closestMonster[1]]
                    selfDistToGoal=self.wave[self.x][self.y]
                    if selfDistToGoal < monsterDistToGoal:
                        self.goForwards=True
                    else:
                        self.goForwards=False

                #going toward the exit
                if self.goForwards:
                    move, bomb = self.abSearch(wrld, self.depthMax, float('-inf'), float('inf'))
                    # Move
                    dx, dy = move
                    self.move(dx, dy)
                    self.moves.append((dx,dy))
                    return
                
                #if going back go back start
                if (self.x,self.y)!=self.start:# self.wave[self.x][self.y]>5:
                    path = self.astar(wrld, [self.x, self.y], self.start,False)
                    if path:
                        nextPoint = path.pop(1)
                        dx,dy = nextPoint[0] - self.x, nextPoint[1] - self.y
                        self.move(dx, dy)
                        self.moves.append((dx,dy))
         
                # if at start stall
                else:
                    pass

    def abSearch(self, wrld, depth, alpha, beta):
        """
        Performs an alpha-beta search to find the best action.

        Args:
            wrld (World): The current game world.
            depth (int): The depth of the search tree.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.

        Returns:
            Action: The optimal action to take.
        """
        # Get best action
        v = self.maxValue(wrld, depth, alpha, beta)
        return self.optimalAction
    
    def maxValue(self, wrld, depth, alpha, beta):
            """
            Calculates the maximum value for the AI character using the minimax algorithm with alpha-beta pruning.

            Parameters:
            - wrld: The current state of the world.
            - depth: The current depth of the search.
            - alpha: The alpha value for alpha-beta pruning.
            - beta: The beta value for alpha-beta pruning.

            Returns:
            - The maximum value for the AI character.
            """
            # Edge cases
            if wrld == None:
                return float('-inf')
            if self.terminal(wrld, depth):
                return self.utility(wrld)
            
            v = float('-inf')
            
            for action in self.actions(wrld):
                v = max(v, self.minValue(self.result(wrld, action), depth-1, alpha, beta))
                if v > alpha and depth == self.depthMax:
                    # Save optimal action
                    self.optimalAction = action
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
    
    def minValue(self, wrld, depth, alpha, beta):
        """
        Calculates the minimum value (the monster's action) in a given game world state.

        Args:
            wrld (GameWorld): The current game world state.
            depth (int): The current depth of the search.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.

        Returns:
            float: The minimum utility value for the AI character.

        """
        if wrld == None:
            return float('-inf')
        if self.terminal(wrld, depth):
            return self.utility(wrld)
        v = float('inf')
        for action in self.monsterActions(wrld):
            v = min(v, self.maxValue(self.result(wrld, action), depth-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def actions(self, wrld):
        """
        Generates a list of possible actions for the AI character based on the current game world state.

        Args:
            wrld (World): The current game world state.

        Returns:
            list: A list of tuples representing the possible actions. Each tuple consists of a movement direction (dx, dy)
            and a flag indicating whether to place a bomb (1) or not (0).
        """
        try:
            path = self.astar(wrld, [wrld.me(self).x, wrld.me(self).y], self.findExit(wrld))
            nextPoint = path.pop(1)
            dx, dy = nextPoint[0] - self.x, nextPoint[1] - self.y
            actions = [(dx, dy), (-dx, -dy),(0,0)]
            monsters=self.findMonsters(wrld)
            if monsters:
                monDist=min([len(self.astar(wrld,monster,[wrld.me(self).x, wrld.me(self).y])) for monster in monsters])
                if monsters and monDist<=3:
                    actions=self.getNeighbors(wrld,(wrld.me(self).x, wrld.me(self).y))
                    actions.append((0,0))
            new_actions = []
            for action in actions:
                new_actions.append((action,0))

            if self.findBomb(wrld) is None:
                for i in range(len(actions)):
                    new_actions.append((actions[i], 1))
            return new_actions
        except:
            return [(1,1),(-1,-1),(1,-1),(-1,1)]
    
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
        try:
            x, y = action[0]
            bomb = action[1]
            newWorld = wrld.from_world(wrld)
            if bomb == 1:
                newWorld.me(self).place_bomb()
            newWorld.me(self).move(x, y)
            nextWorld, _ = newWorld.next()
            return nextWorld
        except:
            return None
    
    def monsterActions(self, wrld):
        """
        Perform actions for the monster character.

        Args:
            wrld: The current game world.

        Returns:
            A list of tuples representing the monster's actions. Each tuple contains:
            - A movement direction as a tuple (dx, dy)
            - A bomb placement indicator (0 for no bomb, 1 for bomb placement)
        """
        # Pretend no moves for speed
        return [((0,0),0)]
    
    def terminal(self, wrld, depth):
            """
            Checks if the current state is a terminal state.

            Parameters:
                wrld (World): The current game world.
                depth (int): The current depth of the search.

            Returns:
                bool: True if the state is terminal, False otherwise.
            """
            # Check terminal conditions
            if depth == 0 or wrld.exit_at(self.x, self.y) or wrld.monsters_at(self.x, self.y) or wrld.explosion_at(self.x, self.y):
                return True
    
    def utility(self, wrld):
        """
        Calculate the utility of the world for the AI character.

        Parameters:
        - wrld (World): The current state of the world.

        Returns:
        - float: The utility value of the world.
        """
        charx, chary = self.findChar(wrld)
        if charx == -1 and chary == -1:
            return float('-inf')
        monsterCost = 0
        bombPoints = 0
        bomb = self.findBomb(wrld)
        monsters = self.findMonsters(wrld)
        selfDistToGoal = self.wave[charx][chary]
        monDist = float('inf')
        if wrld.explosion_at(charx, chary):
            return float('-inf')
        if monsters:
            for monster in monsters:
                dist = len(self.astar(wrld, (charx, chary), monster))
                if dist <= 3:
                    monsterCost += (2 * (4 - dist)) ** 2
                if dist < monDist:
                    monDist = dist
                monsterCost += 50
                if bomb:
                    bombDist = self.openDist(bomb, monster)
                    if bombDist <= 8:
                        bombPoints += 50
            monsterDistToGoal = min([self.wave[monster[0]][monster[1]] for monster in monsters])

            if selfDistToGoal > monsterDistToGoal and monDist <= 2:
                monsterCost *= 4

            elif selfDistToGoal < monsterDistToGoal - 1:
                monsterCost -= 10
                return 7000
        return wrld.time - selfDistToGoal - monsterCost + bombPoints
        
    def astar(self, wrld, start, goal, withMonster=True):
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
        bombs = self.findBomb(wrld)
        while not found and not pq.empty():
            element = pq.get()
            exploring = element[0]
            g = element[2]
            explored[exploring] = element
            if exploring == tuple(goal):
                found = True
                break

            neighbors = self.getNeighbors(wrld, exploring)
            for neighbor in neighbors:
                if explored.get(neighbor) is None or explored.get(neighbor)[2] > g + 1:
                    monstersCost = 0
                    if neighbor == bombs:
                        monstersCost += 1000
                    if withMonster:
                        monsters = self.findMonsters(wrld)
                        for monster in monsters:
                            dist = self.heuristic((neighbor[0], neighbor[1]), monster)
                            if dist <= 3:
                                monstersCost += 2 * (4 - dist)
                    f = g + 1 + self.heuristic(neighbor, goal) + monstersCost
                    pq.put((neighbor, exploring, g + 1), f)
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
        return path

    #Helper function to return the walkable neighbors 
    def getNeighbors(self, wrld, cell):
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

        if celly < rows - 1:
            if wrld.wall_at(cellx, celly + 1) == 0 and wrld.explosion_at(cellx, celly + 1) is None and not self.checkExplode(
                    wrld, self.findBomb(wrld), (cellx, celly + 1)):
                neighbors.append((cellx, celly + 1))
        if cellx < cols - 1:
            if wrld.wall_at(cellx + 1, celly) == 0 and wrld.explosion_at(cellx + 1, celly) is None and not self.checkExplode(
                    wrld, self.findBomb(wrld), (cellx + 1, celly)):
                neighbors.append((cellx + 1, celly))
            if celly > 0:
                if wrld.wall_at(cellx + 1, celly - 1) == 0 and wrld.explosion_at(cellx + 1, celly - 1) is None and not self.checkExplode(
                        wrld, self.findBomb(wrld), (cellx + 1, celly - 1)):
                    neighbors.append((cellx + 1, celly - 1))
                if celly < rows - 1:
                    if wrld.wall_at(cellx + 1, celly + 1) == 0 and wrld.explosion_at(cellx + 1, celly + 1) is None and not self.checkExplode(
                            wrld, self.findBomb(wrld), (cellx + 1, celly + 1)):
                        neighbors.append((cellx + 1, celly + 1))
        if celly > 0:
            if wrld.wall_at(cellx, celly - 1) == 0 and wrld.explosion_at(cellx, celly - 1) is None and not self.checkExplode(
                    wrld, self.findBomb(wrld), (cellx, celly - 1)):
                neighbors.append((cellx, celly - 1))
        if cellx > 0:
            if wrld.wall_at(cellx - 1, celly) == 0 and wrld.explosion_at(cellx - 1, celly) is None and not self.checkExplode(
                    wrld, self.findBomb(wrld), (cellx - 1, celly)):
                neighbors.append((cellx - 1, celly))
            if celly > 0:
                if wrld.wall_at(cellx - 1, celly - 1) == 0 and wrld.explosion_at(cellx - 1, celly - 1) is None and not self.checkExplode(
                        wrld, self.findBomb(wrld), (cellx - 1, celly - 1)):
                    neighbors.append((cellx - 1, celly - 1))
                if celly < rows - 1:
                    if wrld.wall_at(cellx - 1, celly + 1) == 0 and wrld.explosion_at(cellx - 1, celly + 1) is None and not self.checkExplode(
                            wrld, self.findBomb(wrld), (cellx - 1, celly + 1)):
                        neighbors.append((cellx - 1, celly + 1))
        return neighbors
    def getNeighbors(self,wrld, cell):
        cellx=cell[0]
        celly=cell[1]
        neighbors=[]
        rows, cols = wrld.height(), wrld.width()
        
        if cellx<0 or cellx>=cols or celly<0 or celly>=rows:
            return neighbors

        if wrld.wall_at(cellx,celly)==1:
            return neighbors
        
        if celly<rows-1:
            if wrld.wall_at(cellx,celly+1)==0 and wrld.explosion_at(cellx,celly+1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx,celly+1)):
                neighbors.append((cellx,celly+1))
        if cellx<cols-1:
            if wrld.wall_at(cellx+1,celly)==0 and wrld.explosion_at(cellx+1,celly) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx+1,celly)):
                neighbors.append((cellx+1,celly))
            if celly>0:
                if wrld.wall_at(cellx+1,celly-1)==0 and wrld.explosion_at(cellx+1,celly-1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx+1,celly-1)):
                    neighbors.append((cellx+1,celly-1)) 
            if celly<rows-1:
                if wrld.wall_at(cellx+1,celly+1)==0 and wrld.explosion_at(cellx+1,celly+1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx+1,celly+1)):
                    neighbors.append((cellx+1,celly+1))
        if celly>0:
            if wrld.wall_at(cellx,celly-1)==0 and wrld.explosion_at(cellx,celly-1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx,celly-1)):
                neighbors.append((cellx,celly-1))
        if cellx>0:
            if wrld.wall_at(cellx-1,celly)==0 and wrld.explosion_at(cellx-1,celly) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx-1,celly)):
                neighbors.append((cellx-1,celly))
            if celly>0:
                if wrld.wall_at(cellx-1,celly-1)==0 and wrld.explosion_at(cellx-1,celly-1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx-1,celly-1)):
                    neighbors.append((cellx-1,celly-1)) 
            if celly<rows-1:
                if wrld.wall_at(cellx-1,celly+1)==0 and wrld.explosion_at(cellx-1,celly+1) is None and not self.checkExplode(wrld,self.findBomb(wrld),(cellx-1,celly+1)):
                    neighbors.append((cellx-1,celly+1))
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
    
    
    def generateWavefront(self, wrld, start):
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
        queue = [(0, start[0], start[1])]
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1),(0, 1), (1, 0), (0, -1), (-1, 0)]
        while queue:
            cost, curCol,curRow = heappop(queue)
            for direction in directions:
                newCol, newRow = curCol + direction[0], curRow + direction[1]
                if (0 <= newRow < rows) and (0 <= newCol < cols):# and (not wrld.wall_at(newCol, newRow)):
                    if wrld.wall_at(newCol, newRow)==0:
                        newCost = cost + 1
                        if newCost < wavefront[newCol][newRow]:
                            wavefront[newCol][newRow] = newCost
                            heappush(queue, (newCost, newCol, newRow))
        return wavefront
    

    def heuristic(self, p1, p2):
        """
        Calculate the heuristic value between two points.

        Args:
            p1 (tuple): The coordinates of the first point.
            p2 (tuple): The coordinates of the second point.

        Returns:
            int: The heuristic value between the two points.
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
            return False
        if wrld.bomb_at(bomb[0], bomb[1]).timer <= 1:
            if (abs(tile[0] - bomb[0]) <= 5 and tile[1] == bomb[1]) or (abs(tile[1] - bomb[1]) <= 5 and tile[0] == bomb[0]):
                return True