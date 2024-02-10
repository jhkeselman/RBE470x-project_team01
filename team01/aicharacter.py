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

# class State(Enum):
#     SEARCH = 0
#     EXIT = 1

class AICharacter(CharacterEntity):
    
    # curState = 0
    # name, avatar, x, y = "", "", 0, 0
    # def __init__(self, name, avatar, x, y):
    #     self.name = name
    #     self.avatar = avatar
    #     self.x = x
    #     self.y = y
    #     self.curState = State.SEARCH

    optimalAction = ()
    depthMax = 3
    wave=None

    def do(self, wrld):
        
        # if self.curState == self.State.SEARCH:
        for x in range(wrld.width()):
            self.set_cell_color(x, 0, Fore.RED + Back.GREEN)
            
        exitX, exitY = self.findExit(wrld)
        if self.wave is None:
            self.wave=self.generateWavefront(wrld, [exitX, exitY])
        path = self.astar(wrld, [self.x, self.y], [exitX, exitY])
        # if len(path) != 0:
        #     self.curState = self.State.EXIT
        # # # # # # # # # # #     
        # if self.curState == self.State.EXIT:
        if path:
            # nextPoint = path.pop(1)
            # dx, dy = nextPoint[0] - self.x, nextPoint[1] - self.y
            # self.move(dx, dy)
            dx, dy = self.abSearch(wrld, self.depthMax, float('-inf'), float('inf'))
            self.move(dx, dy)

    def abSearch(self, wrld, depth, alpha, beta):
        v = self.maxValue(wrld, depth, alpha, beta)
        return self.optimalAction
    
    def maxValue(self, wrld, depth, alpha, beta):
        if depth == 0:
            return self.utility(wrld)
        v = float('-inf')
        for action in self.actions(wrld):
            v = max(v, self.minValue(self.result(wrld, action), depth-1, alpha, beta))
            if v > alpha and depth == self.depthMax:
                self.optimalAction = action
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, wrld, depth, alpha, beta):
        if depth == 0:
            return self.utility(wrld)
        v = float('inf')
        for action in self.monsterActions(wrld):
            v = min(v, self.maxValue(self.result(wrld, action), depth-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def actions(self, wrld):
        path = self.astar(wrld, [wrld.me(self).x, wrld.me(self).y], self.findExit(wrld))
        nextPoint = path.pop(1)
        dx, dy = nextPoint[0] - self.x, nextPoint[1] - self.y
        actions = [(dx, dy), (-dx, -dy)]
        return actions
    
    def result(self, wrld, action):
        newWorld = wrld.from_world(wrld)
        newWorld.me(self).move(action[0], action[1])
        ex, ey = self.findExit(wrld)
        if newWorld.me(self).x+action[0] == ex and  newWorld.me(self).y+action[1] == ey:
            return wrld
        nextWorld, _ = newWorld.next()
        return nextWorld
    
    def monsterActions(self, wrld):
        return [(0,0)]
    
    def terminal(self, wrld, depth):
        if depth == 0 and wrld.exitAt(self.x, self.y) and wrld.monsters_at(self.x, self.y):
            return True
    
    def utility(self, wrld):
        charx, chary = self.findChar(wrld)
        return wrld.time - self.wave[charx][chary]
        
    def astar(self, wrld, start, goal):
        path = []
        found = False

        pq=PriorityQueue() #
        pq.put((tuple(start),None,0),0)
        explored={}#dict of everything being added as the key and the node that added them (the item) for easy checking to prevent re adding. easy to retrace the path with
        # print(start,goal)
        
        while not found and not pq.empty():#if there is nothing left in the q there is nothing left to explore therefore no possible path to find
            element=pq.get()#pulling the first item added to the q which will in practice be the lowest level for bfs exploration
            exploring=element[0]
            g=element[2]
            explored[exploring] = element
            if exploring == tuple(goal):#exploring the goal exit condition
                found=True
                # print('found goal')
                break
            
            neighbors=self.getNeighbor(wrld,exploring)
            for neighbor in neighbors:
                if explored.get(neighbor) is None or explored.get(neighbor)[2]>g+1:
                    monstersCost=0
                    monsters=self.findMonsters(wrld)
                    for monster in monsters:
                        dist=self.heuristic((neighbor[0], neighbor[1]), monster)
                        if dist<=3:
                            monstersCost+=2*(4-dist)
                    f=g+1+self.heuristic(neighbor,goal)+monstersCost#heuristic is the manhattan distance and is used cause this is A*
                
                    pq.put((neighbor,exploring,g+1),f)
            # print(pq.get_queue())
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
        #     print(f"It takes {steps} steps to find a path using A*")
        # else:
        #     print("No path found")
        return path

    #Helper function to return the walkable neighbors 
    def getNeighbor(self,wrld, cell):
        cellx=cell[0]
        celly=cell[1]
        neighbors=[]
        rows, cols = wrld.height(), wrld.width()
        
        if cellx<0 or cellx>=cols or celly<0 or celly>=rows:
            return neighbors
        # print(cellx,celly)
        if wrld.wall_at(cellx,celly)==1:
            return neighbors
        
        if celly<rows-1:
            if wrld.wall_at(cellx,celly+1)==0:
                neighbors.append((cellx,celly+1))
        if cellx<cols-1:
            if wrld.wall_at(cellx+1,celly)==0:
                neighbors.append((cellx+1,celly))
            if celly>0:
                if wrld.wall_at(cellx+1,celly-1)==0:
                    neighbors.append((cellx+1,celly-1)) 
            if celly<rows-1:
                if wrld.wall_at(cellx+1,celly+1)==0:
                    neighbors.append((cellx+1,celly+1))
        if celly>0:
            if wrld.wall_at(cellx,celly-1)==0:
                neighbors.append((cellx,celly-1))
        if cellx>0:
            if wrld.wall_at(cellx-1,celly)==0:
                neighbors.append((cellx-1,celly))
            if celly>0:
                if wrld.wall_at(cellx-1,celly-1)==0:
                    neighbors.append((cellx-1,celly-1)) 
            if celly<rows-1:
                if wrld.wall_at(cellx-1,celly+1)==0:
                    neighbors.append((cellx-1,celly+1))

        return neighbors


    #Helper function to reconstruct the path from the explored dictionary
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
    

    def heuristic(self,p1, p2):
        return sum([abs(p2[0]-p1[0]),abs(p2[1]-p1[1])])
        
    def findExit(self,wrld):
        exitY, exitX = 0, 0
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.exit_at(col, row):
                    exitY, exitX = row, col
        return exitX, exitY
    
    def findChar(self,wrld):
        y, x = -1, -1
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.characters_at(col, row):
                    y, x = row, col
        return x, y
    
    def findMonsters(self,wrld):
        monsters=[]
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.monsters_at(col, row):
                    monsters.append((row, col))
        return monsters
    
    def findBomb(self,wrld):
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.monsters_at(col, row):
                    return (col, row)
        return None
