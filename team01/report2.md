# <p style="text-align: center;">RBE470x Project 2 Report</p>
## <p style="text-align: center;">By Josh Keselman, Sam Markwick, and Cam Wian</p>

## 1. Overall Structure of Our Approach
Our character control code is contained within a file called `qlearningcharacter.py`. Our code has built primarily around approximate q-learning, with a few helper functions to assist in the learning process, A* navigation, and a state machine to control the character's actions.

We utilized a state machine to allow our character to switch between an "exploration" state, where q-learning is used to navigate through the map while updating weights to optimize future decisions, and a "goal" state, where the character uses A* to navigate directly to the goal. The character starts in the "exploration" state, and switches to the "goal" state once there is a safe pathway to the goal. 

## 2. Code Breakdown
This section will dive deeper into specific pieces of our code. We will briefly cover the structure of our state machine and our A* implementation, as these have remained largely unchanged from Project 1. We will then discuss our approximate q-learning implementation and the helper functions we used to assist in the learning process, as well as our training environment for q-learning.

### 2.1 Initialization
We modified our `__init__` method to include the initialization of our q-learning weights, as well as the loading of weights from a .csv file if they exist. This allows us to train our character over multiple games, and to save the weights to a file for future use. We're using `numpy` to help manage our weights and the .csv file. If the `weights.csv` file does not exist, we initialize our weights to random values between 0 and 1, and if it does exist, we load the weights from the file, with a small check to ensure that the weights are matched to the features.


### 2.2 State Machine
Like project 1, our state machine was implemented in the `do` function. The state machine is generally structured as follows:
```
if not initialized:
    Initialize character variables including wavefronts to goal with and without walls
Get A* path to goal
If there is a path to goal and no monsters:
    A* to goal
else if there is a path to goal:
    Get shortest distance from the goal to a monster
    If that distance is two greater than distance from character to goal:
        A* to goal

get optimal action using q-learning
if no optimal action or random chance (10% of the time):
    Choose random action
else:
    Choose optimal action from results of q-learning
Take chosen action
Update weights
Save updated weights to .csv file
```
This structure allows the character to navigate to the goal using A* if there is a clear path, and to use q-learning to navigate if there is not. It also uses a static epsilon value to allow for a mix of exploration and exploitation. This method, of balancing between direct pathfinding and q-learning, ensures that the character is extremely efficient on the easiest variants, as well as placing more weighting emphasis on actions taken further from the goal, where monsters and bombs are generally more threatening. 

### 2.3 A* Implementation and Wavefront
Our A* implementation is primarily the same as our implementation from project 1, with few small change. First, we removed our added heuristics to avoid monsters and bombs automatically, as we want q-learning to able to adjust for these. We also added variables to allow for toggling on or off walls, bombs, and monsters. This allows us to use A* to both generate actions and find pure distances between points in the game world. We also included a toggle to switch between 4-connectivity and 8-connectivity actions.

We also implemented a wavefront method, that creates an array representing the costs from the current node to every point in the world. This method also has a toggle to either ignore or account for walls. This method is used to help q-learning find the best action to take, as it allows us to find the distance to the goal from every point in the world, and to find the distance to the goal from every point in the world while avoiding walls.
### 2.4 Q-Learning
### 2.5 Training

## 3. Experimental Evaluation

### 3.1 Variant 1:
### 3.2 Variant 2:
### 3.3 Variant 3:
### 3.4 Variant 4:
### 3.5 Variant 5: