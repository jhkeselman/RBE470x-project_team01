# <p style="text-align: center;">RBE470x Project 1 Report</p>
## <p style="text-align: center;">By Josh Keselman, Sam Markwick, and Cam Wian</p>

## 1. Overall Structure of Our Approach
Our character control code is contained within a file called `aicharacter.py`. Our control code has four basic parts: a "state machine" in `do` method, our A* navigation algorithm, our minimax implementation, and a set of helper functions to locate various objects in the game world. 

Our state machine is designed to switch the character between two general states, one without a bomb present on the map (denoted as *no_bomb*) and one with a bomb present (*bomb*). The character starts in the *no_bomb* state, and each of those states has multiple substates. In *no_bomb*, these substates include a general minimax state, to get our next move based on the current game state, as well as determine if we should place a bomb, and states to navigate directly to the goal if no monsters are present or if the character is closer to the goal than any monsters are. In the *bomb* state, the character has a substate to navigate to the goal if no monsters are present or if the character is closer to the goal than any monsters are, and a substate to backtrack previous moves, allowing the character to retreat to safe areas and wait for the bomb to detonate.

We chose to use A* to navigate the game world, as it is an optimal algorithm for discrete pathfinding problems. Despite not being dynamic, it is very effective at finding optimal paths between two points in a grid. This allowed us to use A* not just to find our own path to and distance from the goal, but also to find the distance between the monsters and the goal and the monsters and the character, both helpful for determining the best course of action. Using A* also allowed us to implement a modified priority calculation, that includes not just our heuristic for the goal distance but also an added distance to adjust for nearby monsters.

Our minimax implementation is used to determine the best move for the character to make, based on the current game state. We chose to use minimax, with alpha-beta pruning, due to its relative simplicity to implement and ability to handle large decisions trees through pruning. ...

Finally, we used a set of helper function to find specific items in the world. This included the character's current position, the goal's position, the positions of all monsters, and the position of a bomb, if present. These functions helped us define the exact state of the world, and were designed to allow easy usage with future world states found with `SensedWorld.next()`.
## 2. Code Breakdown
This section will dive deeper into specific pieces of our code. We will discuss the structure of our state machine, our A* implementation, our minimax implementation, and our helper functions.
### 2.1 State Machine
### 2.2 A* Implementation
### 2.3 Minimax Implementation
### 2.4 Helper Functions
Our four *find* functions all follow a similar structure, shown with the following pseudocode:
```
def findItem(self, item):
    loop through world rows
        loop through world columns
            if row and column contain item
                return (row, column)
    return nothing if item not found 
```
This basic structure has slight modifications for each specific item, with each type requiring different return values when not found. The `findMonsters` method uses an array to store the position of monsters, rather than returning directly, as there may be more than one monster present in the world. These functions are also versatile, allowing us to search through either the actual world state or the state of a copied world, which is useful for our minimax implementation.
## 3. Experimental Evaluation
To test our code for project 1, we ran a series of 10 tests for each variant, with different seeds for each test. For each test, we recorded whether or not the character successfully reached the goal, and the score of the character when the run terminated. We then averaged these scores to get a sense of how well our code performed. The results of these tests, as well as some brief analysis, are shown below.

The seeds we used were as follows: 123, 716, 632, 516, 633, 127, 277, 516, 233, 912.
### 3.1 Variant 1
Out of [x] tests, the character successfully reached the goal [y] times. The average score of the character when the run terminated was [z]. This variant should work flawlessly, as it is the simplest of the five variants with no monsters involved.
### 3.2 Variant 2
Out of [x] tests, the character successfully reached the goal [y] times. The average score of the character when the run terminated was [z].
### 3.3 Variant 3
Out of [x] tests, the character successfully reached the goal [y] times. The average score of the character when the run terminated was [z].
### 3.4 Variant 4
Out of [x] tests, the character successfully reached the goal [y] times. The average score of the character when the run terminated was [z].
### 3.5 Variant 5
Out of [x] tests, the character successfully reached the goal [y] times. The average score of the character when the run terminated was [z].