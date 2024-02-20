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

class QLearningCharacter(CharacterEntity):

    weights = []
    

    def do(self, wrld):
        # Commands
        dx, dy = 0, 0
        bomb = False
        goal=self.findExit(wrld)
        wavefront = self.generateWavefront(wrld, goal,True)
        flipwave=np.transpose(wavefront)
        for i in range(len(flipwave)):
            print(flipwave[i])
        
        # Handle input
        for c in input("How would you like to move (w=up,a=left,s=down,d=right,b=bomb)? "):
            if 'w' == c:
                dy -= 1
            if 'a' == c:
                dx -= 1
            if 's' == c:
                dy += 1
            if 'd' == c:
                dx += 1
            if 'b' == c:
                bomb = True
        # Execute commands
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

    def fmonsters(self, point, wavefront):
        return 0
    
    def fexit(self, point, wavefront):
        dis=wavefront[point[0]][point[1]]
        #if dis==float('inf'):
            #circle explore to nearest explorable tile add distance walls as weight.
        return dis
    
    def fexpl(self, point, wavefront):
        return 0

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
            if exploring == tuple(goal):
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
                    f = g + 1 + self.manhattan(neighbor, goal) + penalty
                    pq.put((neighbor, exploring, g + 1), f)
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
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
                    if not withBomb or not self.checkExplode(wrld, self.findBomb(wrld), newCell, turns):
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
    
    def generateWavefront(self, wrld, start,throughWalls=False):
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
            # print(queue.get_queue())
            element = queue.get()
            # print("element",element)
            cost = element[0]
            curCol, curRow = element[1]
            wavefront[curCol][curRow] = cost
            for direction in directions:
                newCol, newRow = curCol + direction[0], curRow + direction[1]
                if (0 <= newRow < rows) and (0 <= newCol < cols):# and (not wrld.wall_at(newCol, newRow)):
                    # print("newCol,newRow",newCol,newRow)
                    if wrld.wall_at(newCol, newRow)==0:
                        newCost = cost + 1
                        if newCost < wavefront[newCol][newRow]:
                            # wavefront[newCol][newRow] = newCost
                            queue.put((newCost, (newCol, newRow)), newCost)
                    elif throughWalls:
                        # print("through walls",newCol,newRow,wrld.wall_at(newCol, newRow))
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

    def checkExplode(self, wrld, bomb, tile,turns=1):
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
        if wrld.bomb_at(bomb[0], bomb[1]).timer <= turns:
            if (abs(tile[0] - bomb[0]) <= 4 and tile[1] == bomb[1]) or (abs(tile[1] - bomb[1]) <= 4 and tile[0] == bomb[0]):
                return True