import sys
from enum import Enum
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
            # Search for exit
            pass
        if curState == State.EXIT:
            # Move to exit
            pass
        pass
