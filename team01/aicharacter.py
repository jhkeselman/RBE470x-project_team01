import sys
from enum import Enum
from heapq import heappush, heappop
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

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
        exitY, exitX = self.findExit(wrld)
        path = self.astar(wrld, [self.y, self.x], [exitY, exitX])
        # if len(path) != 0:
        #     self.curState = self.State.EXIT
        # # # # # # # # # # #     
        # if self.curState == self.State.EXIT:
        while path:
            nextPoint = path.pop(1)
            dx, dy = nextPoint[1] - self.x, nextPoint[0] - self.y
            print(dx, dy)
            self.move(dx,dy)
            path = self.astar(wrld, (self.y, self.x), (exitY, exitX))
        
        
    def astar(self, wrld, start, goal):
        path = []
        found = False
        rows, cols = wrld.height(), wrld.width()
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        costs = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        parent = [[None for _ in range(cols)] for _ in range(rows)]
        parent[start[0]][start[1]] = (-1,-1)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        queue = [(0 + self.heuristic(start, goal), 0, start[0], start[1])]
        costs[start[0]][start[1]] = 0

        while queue:
            _, cost, curRow, curCol = heappop(queue)
            
            if [curRow, curCol] == goal:
                found = True
                break

            for direction in directions:
                newRow, newCol = curRow + direction[0], curCol + direction[1]
                if 0 <= newRow < rows and 0 <= newCol < cols and not visited[newRow][newCol] and not wrld.wall_at(newCol, newRow):
                    monstersCost=0
                    monsters=self.findMonster(wrld)
                    for monster in monsters:
                        print(monster)
                        dist=self.heuristic((newRow, newCol), monster)
                        if dist<=3:
                            monstersCost+=6-dist
                    newCost = cost + 1+monstersCost 
                    if newCost < costs[newRow][newCol]:
                        costs[newRow][newCol] = newCost
                        prio = newCost + self.heuristic((newRow, newCol), goal)
                        parent[newRow][newCol] = (curRow, curCol)
                        visited[newRow][newCol] = True
                        heappush(queue, (prio, newCost, newRow, newCol))

        if found:
            curR, curC = goal[0], goal[1]
            while curR != -1 and curC != -1:
                path.insert(0,[curR, curC])
                curR, curC = parent[curR][curC][0], parent[curR][curC][1]
        return path
    
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
                    monsters.append((row, col))
        return monsters