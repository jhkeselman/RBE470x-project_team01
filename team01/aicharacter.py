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

    def do(self, wrld):
        # if self.curState == self.State.SEARCH:
        for x in range(wrld.width()):
            self.set_cell_color(x, 0, Fore.RED + Back.GREEN)
            
        exitY, exitX = self.findExit(wrld)
        # path = self.astar(wrld, [self.x, self.y], [exitX, exitY])
        # if len(path) != 0:
        #     self.curState = self.State.EXIT
        # # # # # # # # # # #     
        # if self.curState == self.State.EXIT:
        path=True
        while path:
            path = self.astar(wrld, (self.x, self.y), (exitX, exitY))
            if not path:
                break
            nextPoint = path.pop(1)
            print(nextPoint)
            dx, dy = nextPoint[0] - self.x, nextPoint[1] - self.y
            print(dx, dy)
            self.move(dx,dy)
            break
            
            
    
    def expectimax(self,wrld,start,goal):
        actions = self.getActions(wrld,start)
        # max_val = 0
        # arg_max = 0
        # for action in actions:
        #     val = self.expValue(wrld,start,goal)
        #     if(val > max_val):
        #         max_val = val
        #         arg_max = action
        arg_max = self.argMax(actions,self.result) # not working yet i think, needs restructuring
        return arg_max
    
    def expValue(self,wrld,start,goal):
        # if terminal state return utility
        v = 0
        for action in self.getActions(wrld,start):
            p = 1/len(self.getActions(wrld,start))
            v += p*self.maxValue(self.result(wrld, action, goal))
        return v 
    
    def argMax(args,util_function):
        max_val = 0
        arg_max = None
        for arg in args:
            val = util_function(*arg)
            if(val > max_val):
                max_val = val
                arg_max = arg
        return arg_max

    def maxValue(self,wrld,start, goal):
        # if terminal state return utility
        v = float('-inf')
        for action in self.actions(wrld,start):
            v = max(v, self.expValue(self.result(wrld, action, goal)))
        return v
    
    def result(self,wrld,action,start,goal):
        self.move(action[0],action[1])
        nextWrld = wrld.sensedWorld.next()
        return nextWrld, start, goal
    
    def getActions(self, wrld, start):
        possibleActions = [(1,0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        validActions = []
        for action in possibleActions:
            if wrld.wall_at(action[0], action[1]) == False and self.withinBounds(wrld, action[0]+start[0], action[1]+start[1]):
                validActions.append(action)
        return validActions
    
    def withinBounds(self, wrld, x, y):
        if x < 0 or x >= wrld.width() or y < 0 or y >= wrld.height():
            return False
        return True
        
    def astar(self, wrld, start, goal):
        path = []
        steps = 0
        found = False

        pq=PriorityQueue() #
        pq.put((tuple(start),None,0),0)
        explored={}#dict of everything being added as the key and the node that added them (the item) for easy checking to prevent re adding. easy to retrace the path with
        print(start,goal)
        while not found and not pq.empty():#if there is nothing left in the q there is nothing left to explore therefore no possible path to find
            steps+=1
            element=pq.get()#pulling the first item added to the q which will in practice be the lowest level for bfs exploration
            exploring=element[0]
            g=element[2]
            explored[exploring] = element
            if exploring == tuple(goal):#exploring the goal exit condition
                found=True
                print('found goal')
                break
            
            neighbors=self.getNeighbor(wrld,exploring)
            for neighbor in neighbors:
                if explored.get(neighbor) is None or explored.get(neighbor)[2]>g+1:
                    monstersCost=0
                    monsters=self.findMonster(wrld)
                    for monster in monsters:
                        # print(monster)
                        dist=self.heuristic(neighbor, monster)
                        if dist<=4:
                            monstersCost+=2*(4-dist)
                    f=g+1+self.heuristic(neighbor,goal)+monstersCost#heuristic is the manhattan distance and is used cause this is A*
                
                    pq.put((neighbor,exploring,g+1+monstersCost),f)
            # print(pq.get_queue())
        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
            print(f"It takes {steps} steps to find a path using A*")
        else:
            print("No path found")
        return path

    #Helper function to return the walkable neighbors 
    def getNeighbor(self,wrld, cell):
        cellx=cell[0]
        celly=cell[1]
        neighbors=[]
        rows, cols = wrld.height(), wrld.width()
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

        
        # print(neighbors)
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
            print(path)
            return path

        # path = []
        # found = False
        # rows, cols = wrld.height(), wrld.width()
        # visited = [[False for _ in range(cols)] for _ in range(rows)]
        # costs = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        # parent = [[None for _ in range(cols)] for _ in range(rows)]
        # parent[start[0]][start[1]] = (-1,-1)
        # directions = [(1, 1), (1, -1), (-1, -1), (-1, 1),(0, 1), (1, 0), (0, -1), (-1, 0)]
        # queue = [(0 + self.heuristic(start, goal), 0, start[0], start[1])]
        # costs[start[0]][start[1]] = 0

        # while queue:
        #     _, cost, curRow, curCol = heappop(queue)
            
        #     if [curRow, curCol] == goal:
        #         found = True
        #         break

        #     for direction in directions:
        #         newRow, newCol = curRow + direction[0], curCol + direction[1]
        #         if 0 <= newRow < rows and 0 <= newCol < cols and not visited[newRow][newCol] and not wrld.wall_at(newCol, newRow):
        #             monstersCost=0
        #             monsters=self.findMonster(wrld)
        #             for monster in monsters:
                        
        #                 dist=self.heuristic((newRow, newCol), monster)
        #                 if dist<=3:
        #                     monstersCost+=dist
        #             newCost = cost + 1+monstersCost 
        #             if newCost < costs[newRow][newCol]:
        #                 costs[newRow][newCol] = newCost
        #                 prio = newCost + self.heuristic((newRow, newCol), goal)
        #                 parent[newRow][newCol] = (curRow, curCol)
        #                 visited[newRow][newCol] = True
        #                 heappush(queue, (prio, newCost, newRow, newCol))

        # if found:
        #     curR, curC = goal[0], goal[1]
        #     while curR != -1 and curC != -1:
        #         path.insert(0,[curR, curC])
        #         curR, curC = parent[curR][curC][0], parent[curR][curC][1]
        # return path
    
    def heuristic(self,p1, p2):
        return max(abs(p2[0]-p1[0]),abs(p2[1]-p1[1]))
        
    def findExit(self,wrld):
        exitY, exitX = 0, 0
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.exit_at(col, row):
                    exitY, exitX = row, col
        return exitY, exitX
    
    def findMonster(self,wrld):
        monsters=[]
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.monsters_at(col, row):
                    monsters.append((col,row))
        return monsters