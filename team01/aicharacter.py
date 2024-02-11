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
    depthMax = 7
    wave=None
    moves=[]
    

    def do(self, wrld):
            
        exitX, exitY = self.findExit(wrld)
        if self.wave is None:
            self.wave=self.generateWavefront(wrld, [exitX, exitY])
        path = self.astar(wrld, [self.x, self.y], [exitX, exitY])

        if path:
            # Get best move
            if not self.findBomb(wrld):
                move, bomb = self.abSearch(wrld, self.depthMax, float('-inf'), float('inf'))
                # Move
                dx, dy = move
                self.move(dx, dy)
                self.moves.append((dx,dy))
                # Place bomb
                # self.place_bomb()
                if bomb == 1:
                    self.place_bomb()
                if self.openDist(self.findChar(wrld),self.findMonsters(wrld)[0])<=3:
                    self.place_bomb()
            else:
                dx, dy = self.moves.pop()
                self.move(-dx, -dy)
            


    def abSearch(self, wrld, depth, alpha, beta):
        v = self.maxValue(wrld, depth, alpha, beta)
        
        #print("optimal found: ",self.optimalAction)
        return self.optimalAction
    
    def maxValue(self, wrld, depth, alpha, beta):
        # print("maxValue")
        if wrld == None:
            return float('-inf')
        if self.terminal(wrld, depth):
            return self.utility(wrld)
        v = float('-inf')
        try:
            self.set_cell_color(wrld.me(self).x,wrld.me(self).y, Fore.RED + Back.RED)
        except:
            pass
        for action in self.actions(wrld):
            v = max(v, self.minValue(self.result(wrld, action, 0), depth-1, alpha, beta)) # was maxValue
            # print("value returned: ",v)
            if v > alpha and depth == self.depthMax:
                self.optimalAction = action
                # print("set")
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, wrld, depth, alpha, beta):
        # print("minValue")
        if wrld == None:
            return float('-inf')
        # print("depth: ",depth)
        if self.terminal(wrld, depth):
            return self.utility(wrld)
        v = float('inf')
        # try:
        #     self.set_cell_color(wrld.me(self).x,wrld.me(self).y, Fore.MAGENTA + Back.MAGENTA)
        # except:
        #     pass
        # print(self.monsterActions(wrld))
        for action in self.monsterActions(wrld):
            # print("action: ",action)
            v = min(v, self.maxValue(self.result(wrld, action, 1), depth-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def actions(self, wrld):
        try:
            path = self.astar(wrld, [wrld.me(self).x, wrld.me(self).y], self.findExit(wrld))
            nextPoint = path.pop(1)
            dx, dy = nextPoint[0] - self.x, nextPoint[1] - self.y
            actions = [(dx, dy), (-dx, -dy)]
            monsters=self.findMonsters(wrld)
            if monsters:
                monDist=min([len(self.astar(wrld,monster,[wrld.me(self).x, wrld.me(self).y])) for monster in monsters])
                if monsters and monDist<=3:
                    actions=[(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]

            new_actions = []
            for action in actions:
                new_actions.append((action,0))

            if self.findBomb(wrld) is None:
                for i in range(len(actions)):
                    new_actions.append((actions[i], 1))
            return new_actions
        except Exception as e:
            return [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
    
    def result(self, wrld, action, mode):
        try:
            x,y = action[0]
            bomb = action[1]
            newWorld = wrld.from_world(wrld)
            if bomb == 1:
                newWorld.me(self).place_bomb()
            newWorld.me(self).move(x, y)
            nextWorld, _ = newWorld.next()
            return nextWorld
        except:
            return None
        # move = action[0]
        # place_bomb = action[1]
        # try:
        #     newWorld = wrld.from_world(wrld)
        #     newWorld.me(self).move(move[0], move[1])
        #     if place_bomb == 1:
        #         newWorld.me(self).place_bomb()
        #     try:
        #         if(mode == 0):
        #             for monster in newWorld.monsters.values():
        #                 monster.move(0, 0)
        #         if(mode == 1):
        #             for monster in newWorld.monsters.values():
        #                 monster.move(action[0], action[1])
        #     except:
        #         pass
        #     ex, ey = self.findExit(wrld)
        #     if newWorld.me(self).x + action[0] == ex and newWorld.me(self).y + action[1] == ey:
        #         return wrld
        #     nextWorld, _ = newWorld.next()
        #     return nextWorld
        # except:
        #     return None
    
    def monsterActions(self, wrld):
        # monDist=float('inf')
        # monsters=self.findMonsters(wrld)
        # for monster in monsters:
        #     dist = len(self.astar(wrld,(wrld.me(self).x, wrld.me(self).y), monster))
        #     if dist<monDist:
        #         monDist=dist
        # if monDist<=2:
        #     path=self.astar(wrld,monster,[wrld.me(self).x, wrld.me(self).y])
        #     nextPoint = path.pop(1)
        #     dx, dy = nextPoint[0] - monster[0], nextPoint[1] - monster[1]
        #     return [((dx, dy),0)]
        # else:
        #     return [((1,0),0), ((0,1),0), ((-1,0),0), ((0,-1),0), ((1,1),0), ((-1,-1),0), ((1,-1),0), ((-1,1),0)]
        return [((0,0),0)]
        # charx, chary = self.findChar(wrld)
        # monsters=self.findMonsters(wrld)
        # actions=[]
        # for monster in monsters:
        #     path = self.astar(wrld, monster, (charx, chary))
        #     if path:
        #         nextPoint = path.pop(1)
        #         dx, dy = nextPoint[0] - monster[0], nextPoint[1] - monster[1]
        #         actions.append((dx, dy))
        # return actions
    
    def terminal(self, wrld, depth):
        if depth == 0 or wrld.exit_at(self.x, self.y) or wrld.monsters_at(self.x, self.y) or wrld.explosion_at(self.x, self.y):
            return True
    
    def utility(self, wrld):
        charx, chary = self.findChar(wrld)
        # if charx==-1 and chary==-1:
        #     return float('-inf')
        monsterCost = 0
        bombPoints = 0
        bomb=self.findBomb(wrld)
        monsters = self.findMonsters(wrld)
        selfDistToGoal=self.wave[charx][chary]
        monDist=float('inf')
        if monsters:
            # monDist=min([len(self.astar(wrld,monster,[wrld.me(self).x, wrld.me(self).y])) for monster in monsters])
            for monster in monsters:
                dist = len(self.astar(wrld,(charx, chary), monster))
                if dist <= 3:
                    monsterCost += (2*(4-dist))**2
                if dist<monDist:
                    monDist=dist
                monsterCost+=50
                if bomb:
                    bombDist=self.openDist(bomb,monster)
                    if bombDist<=8:# and (abs(bomb[0]-monster[0])<=2 or abs(bomb[1]-monster[1])<=2):
                        bombPoints+=50
            monsterDistToGoal=min([self.wave[monster[0]][monster[1]] for monster in monsters])
            
            if selfDistToGoal > monsterDistToGoal and monDist<=2:
                monsterCost *=4
                # selfDistToGoal+=20
            elif selfDistToGoal < monsterDistToGoal:
                monsterCost -= 10
        return wrld.time - selfDistToGoal - monsterCost + bombPoints
        
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
    
    def openDist(self,p1,p2):
        return max(abs(p2[0]-p1[0]),abs(p2[1]-p1[1]))
        
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
                    monsters.append((col, row))
        return monsters
    
    def findBomb(self,wrld):
        for row in range(wrld.height()):
            for col in range(wrld.width()):
                if wrld.bomb_at(col, row):
                    return (col, row)
        return None
