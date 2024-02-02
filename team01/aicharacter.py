import sys
from enum import Enum
from heapq import heappush, heappop
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

class AICharacter(CharacterEntity):

    def do(self, wrld):
        class State(Enum):
            SEARCH = 0
            EXIT = 1
        curState = State.SEARCH

        if curState == State.SEARCH:
            exitRow, exitCol = 10, 10
            for row in range(wrld.height()):
                for col in range(wrld.width()):
                    if wrld.exit_at(col, row):
                        exitRow, exitCol = row, col
            print(exitRow, exitCol)
            dx, dy = 0, 0
            bomb = False
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
            self.move(dx, dy)
        if curState == State.EXIT:
            # Move to exit
            pass
        pass