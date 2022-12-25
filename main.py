#   PROJECT SOKOBAN: GROUP 22 - BOX,BOX!
#   Aarush Narang             (2010110009)      
#   Moksha Chawla             (2010110402)                 
#   Prerna Chakraborty        (2010110485)
#   Sarthak Bhatnagar         (2010110896)
#   Shashwat Prakash          (2010111108)

import os
import datetime, time
import collections
import heapq
import math

infinity = float('inf')

# Abstract interfaces for the search problems and the search algorithms.

class Search_Problem:
    def start(self): raise NotImplementedError("Override")
    def goalp(self, state): raise NotImplementedError("Override")
    def expand(self, state): raise NotImplementedError("Override")

class Search_Algorithm:
    def solve(self, problem): raise NotImplementedError("Override")

# A* search algorithm
class AStar(Search_Algorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0
        self.finalCosts = collections.defaultdict(lambda:infinity) # Avoid KeyError, and make it callable

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.start()
        closed = []
        self.finalCosts[startState] = 0
        frontier.update(startState, self.heuristic(startState))

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, estimatedCost = frontier.removeMin()

            if state == None: break

            pastCost = self.finalCosts[state]


            self.numStatesExplored += 1

            # Check if we've reached an end state; if so, extract solution.
            if problem.goalp(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.expand(state):
                if newState in closed:
                    continue
                newPastCost = pastCost + cost
                self.finalCosts[newState] = min(newPastCost,self.finalCosts[newState])

                if frontier.update(newState, newPastCost + self.heuristic(newState)):
                # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
                    


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -1*infinity
        self.heap = []
        self.priorities = {}  # Map from state to priority

    
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)

        return (None, None) # Nothing left...


class SokobanState:
    # player: A 2 element tuple with x and y coordinates of the player
    # boxes: A list of A 2 element tuple with x and y coordinates of the boxes
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # Variables to avoid duplicated computation
        self.adj = {}
        self.dead = None
        self.solved = None
        #self.get_area(map)

    # Override the basic functionality of == operator (dunder method)
    def __eq__(self, other):
        return ((type(self) == type(other)) and (self.data == other.data))

    # Implement the functionality of the < operator (magic method)
    def __lt__(self, other):
        return self.data < other.data 
    
    # Returns the hash value of an immutable object (tuple, dict...)
    def __hash__(self):
        return hash(self.data)
    
    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    # 
    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    # def print_theo_state(self, s):
    #     for row in range(len(self.theoretical_map)):
    #         for col in range(len(self.theoretical_map[row])):
    #             target = self.theoretical_map[row][col].target
    #             box = (row, col) in s.boxes()
    #             player = (row, col) == s.player()
    #             if box and target:
    #                 print('*', end='')
    #             elif player and target:
    #                 print('@', end='')
    #             elif target:
    #                 print('.', end='')
    #             #elif box:
    #                 #print('$', end='')
    #             elif player:
    #                 print('@', end='')
    #             elif self.theoretical_map[row][col].wall:
    #                 print('#', end='')
    #             else:
    #                 print(' ', end='')
    #         print()

    # def get_area(self,map):
    #     for row in range(len(map)):
    #         for col in range(len(map[row])):
    #             target = map[row][col].target
    #             box = (row, col) in self.boxes()
    #             player = (row, col) == self.player()
    #             if box and target:
    #                 self.areamatrix[-1].append(-1)
    #             elif player and target:
    #                 self.areamatrix[-1].append(1)
    #             elif target:
    #                 self.areamatrix[-1].append(0)
    #             elif box:
    #                 self.areamatrix[-1].append(-1)
    #             elif player:
    #                 self.areamatrix[-1].append(1)
    #             elif map[row][col].wall:
    #                 self.areamatrix[-1].append(2)
    #             else:
    #                 self.areamatrix[-1].append(0)
    #         self.areamatrix.append([])
    #     self.theostate_ff()
    #     return 
            

    # def theostate_ff(self,x=-1,y=-1):
    #     if x == -1 and y == -1:
    #         player = self.player()
    #         x = player[0]
    #         y = player[1]

    #     if self.areamatrix[x][y-1] == 0:
    #         self.areamatrix[x][y-1] = 1
    #         self.theostate_ff(x,y-1)
    #     elif self.areamatrix[x][y+1] == 0:
    #         self.areamatrix[x][y+1] = 1
    #         self.theostate_ff(x,y+1)
    #     elif self.areamatrix[x-1][y] == 0:
    #         self.areamatrix[x-1][y] = 1
    #         self.theostate_ff(x-1,y)
    #     elif self.areamatrix[x+1][y] == 0:
    #         self.areamatrix[x+1][y] = 1
    #         self.theostate_ff(x+1,y)

        
    def Cornered_Box(self, map, box, targets, all_boxes):

        def row_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[1] + direction
                while not map[box[0]][index].wall:
                    if map[box[0] + offset][index].floor:
                        return None
                    elif map[box[0]][index].target:
                        target_count += 1
                    elif (box[0], index) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        def column_is_trap(offset):
            target_count = 0
            box_count = 1
            for direction in [-1, 1]:
                index = box[0] + direction
                while not map[index][box[1]].wall:
                    if map[index][box[1] + offset].floor:
                        return None
                    elif map[index][box[1]].target:
                        target_count += 1
                    elif (index, box[1]) in all_boxes:
                        box_count += 1
                    index += direction

            if box_count > target_count:
                return True
            return None

        # Corners by Walls
        if box not in targets:
            if map[box[0] - 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] - 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True

        # Expanded corners
            if map[box[0] - 1][box[1]].wall:
                if row_is_trap(offset=-1):
                    return True
            elif map[box[0] + 1][box[1]].wall:
                if row_is_trap(offset=1):
                    return True
            elif map[box[0]][box[1] - 1].wall:
                if column_is_trap(offset=-1):
                    return True
            elif map[box[0]][box[1] + 1].wall:
                if column_is_trap(offset=1):
                    return True
        return None
    def Cornered_Box2(self, map, box, targets, all_boxes):
        rotate_Pattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
        flip_Pattern = [[2,1,0,5,4,3,8,7,6],
                        [0,3,6,1,4,7,2,5,8],
                        [2,1,0,5,4,3,8,7,6][::-1],
                        [0,3,6,1,4,7,2,5,8][::-1]]
        all_Pattern = rotate_Pattern + flip_Pattern
        posofBox = all_boxes
        for box in posofBox:
            if box not in targets:
                board = [[box[0] - 1, box[1] - 1], [box[0] - 1, box[1]], [box[0] - 1, box[1] + 1], 
                        [box[0], box[1] - 1], [box[0], box[1]], [box[0], box[1] + 1], 
                        [box[0] + 1, box[1] - 1], [box[0] + 1, box[1]], [box[0] + 1, box[1] + 1]]
                for pattern in all_Pattern:
                    newBoard = [board[i] for i in pattern]

                    if map[newBoard[1][0]][newBoard[1][1]].wall and map[newBoard[5][0]][newBoard[5][1]].wall: 
                        return True
                    elif newBoard[1] in posofBox and map[newBoard[2][0]][newBoard[2][1]].wall and map[newBoard[5][0]][newBoard[5][1]].wall:
                        return True
                    elif newBoard[1] in posofBox and map[newBoard[2][0]][newBoard[2][1]].wall and newBoard[5] in posofBox: 
                        return True
                    elif newBoard[1] in posofBox and newBoard[2] in posofBox and newBoard[5] in posofBox: 
                        return True
                    elif newBoard[1] in posofBox and newBoard[6] in posofBox and map[newBoard[2][0]][newBoard[2][1]].wall and map[newBoard[3][0]][newBoard[3][1]].wall and [newBoard[8][0]][newBoard[8][1]].wall: 
                        return True
        return False
    def adj_box(self, box, all_boxes):
        adj = []
        for i in all_boxes:
            if box[0] - 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[0] + 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[1] - 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
            elif box[1] + 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
        return adj

    def box_is_trapped(self, map, box, targets, all_boxes):
        if self.Cornered_Box(map, box, targets, all_boxes):
            return True

        adj_boxes = self.adj_box(box, all_boxes)
        for i in adj_boxes:
            if box not in targets and i not in targets:
                if i['direction'] == 'vertical':
                    if map[box[0]][box[1] - 1].wall and map[i['box'][0]][i['box'][1] - 1].wall:
                        return True
                    elif map[box[0]][box[1] + 1].wall and map[i['box'][0]][i['box'][1] + 1].wall:
                        return True
                if i['direction'] == 'horizontal':
                    if map[box[0] - 1][box[1]].wall and map[i['box'][0] - 1][i['box'][1]].wall:
                        return True
                    elif map[box[0] + 1][box[1]].wall and map[i['box'][0] + 1][i['box'][1]].wall:
                        return True
        return None

    
    def box_in_tunnel(self, map, box, all_boxes):
        temp_boxes = self.data[1:]
        for box in list(temp_boxes):
            if map[box[0]+1][box[0]].wall and map[box[0]-1][box[0]].wall:
                return False
            elif map[box[0]][box[0]+1].wall and map[box[0]][box[0]-1].wall:
                return True

    def deadp(self, problem):
        temp_boxes = self.data[1:]
        for box in list(temp_boxes):
            if self.box_is_trapped(problem.map, box, problem.targets, temp_boxes):
                self.dead = True
        return self.dead


class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target


def parse_move(move):
    if move == 'u':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    elif move == 'l':
        return (0, -1)
    elif move == 'r':
        return (0, 1)
    raise Exception('Error: Invalid move.')


class SokobanProblem(Search_Problem):
    # Sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        #self.theoretical_map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0, 0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)
        #self.parse_theoretical_map(theoretical_map)

    # Convert the input string into game map
    # ’ ’ (white space) - Floor
    # @ - Player
    # + - Player on goal
    # # - Wall
    # $ - Stone/Crate/Box
    # . - Goal
    # * - Stone/Box/Crate on Goal


    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map) - 1, len(self.map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def parse_theoretical_map(self, input_str):
        coordinates = lambda: (len(self.theoretical_map) - 1, len(self.theoretical_map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.theoretical_map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.theoretical_map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.theoretical_map[-1].append(MapTile(floor=True))
                #self.init_player = coordinates()
            elif c == '+':
                self.theoretical_map[-1].append(MapTile(floor=True, target=True))
                #self.init_player = coordinates()
                #self.targets.append(coordinates())
            elif c == '$':
                self.theoretical_map[-1].append(MapTile(wall=True))
                #self.init_boxes.append(coordinates())
            elif c == '*':
                self.theoretical_map[-1].append(MapTile(wall=True))
                #self.init_boxes.append(coordinates())
                #self.targets.append(coordinates())
            elif c == '.':
                self.theoretical_map[-1].append(MapTile(floor=True, target=True))
                #self.targets.append(coordinates())
            elif c == '\n':
                self.theoretical_map.append([])
        #assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        #self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print('*', end='')
                elif player and target:
                    print('@', end='')
                elif target:
                    print('.', end='')
                elif box:
                    print('$', end='')
                elif player:
                    print('@', end='')
                elif self.map[row][col].wall:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()

    def print_theo_state(self, s):
        for row in range(len(self.theoretical_map)):
            for col in range(len(self.theoretical_map[row])):
                target = self.theoretical_map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print('*', end='')
                elif player and target:
                    print('@', end='')
                elif target:
                    print('.', end='')
                #elif box:
                    #print('$', end='')
                elif player:
                    print('@', end='')
                elif self.theoretical_map[row][col].wall:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()

    
    # Check for deadlocks
    def dead_lock(self, s):
        if not self.dead_detection:
            return False

        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def flood_fill(self, problem, matrix, path_list, current_path, x, y):
        # matrix = map
        box_pos = problem.data[1:]
        # stop clause - not reinvoking for when there's floor and a box position and a wall.
        if matrix[x][y].floor and not matrix[x][y].visited:
            matrix[x][y].visited = True

            # checks future pos is box
            if (x - 1, y) in box_pos:
                if not matrix[x - 2][y].wall and (x - 2, y) not in box_pos:
                    path_list.append(current_path + 'u')
            if (x + 1, y) in box_pos:
                if not matrix[x + 2][y].wall and (x + 2, y) not in box_pos:
                    path_list.append(current_path + 'd')
            if (x, y - 1) in box_pos:
                if not matrix[x][y - 2].wall and (x, y - 2) not in box_pos:
                    path_list.append(current_path + 'l')
            if (x, y + 1) in box_pos:
                if not matrix[x][y + 2].wall and (x, y + 2) not in box_pos:
                    path_list.append(current_path + 'r')

            # checks each direction if visited, if wall, if box
            if not matrix[x - 1][y].wall and (x - 1, y) not in box_pos and not matrix[x - 1][y].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'u', x - 1, y)
            if not matrix[x + 1][y].wall and (x + 1, y) not in box_pos and not matrix[x + 1][y].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'd', x + 1, y)
            if not matrix[x][y - 1].wall and (x, y - 1) not in box_pos and not matrix[x][y - 1].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'l', x, y - 1)
            if not matrix[x][y + 1].wall and (x, y + 1) not in box_pos and not matrix[x][y + 1].visited:
                self.flood_fill(problem, matrix, path_list, current_path + 'r', x, y + 1)
            return path_list
        return path_list

    def get_position_from_path(self, player, path):
        for move in path:
            if move == 'u':
                player = (player[0] - 1, player[1])
            elif move == 'd':
                player = (player[0] + 1, player[1])
            elif move == 'l':
                player = (player[0], player[1] - 1)
            elif move == 'r':
                player = (player[0], player[1] + 1)
        return player

    def expand(self, s):
        if self.dead_lock(s):
            return []

        for i in self.map:
            for j in i:
                j.visited = False
                
        path_list = self.flood_fill(s, self.map, list(), '', s.data[0][0], s.data[0][1])

        new_states = []
        for path in path_list:
            # Move player
            new_player = self.get_position_from_path(s.data[0], path)

            # Move the box
            box_index = list(s.data[1:]).index(new_player)
            new_boxes = list(s.data[1:])
            if path[-1] == 'u':
                new_boxes[box_index] = (new_boxes[box_index][0] - 1, new_boxes[box_index][1])
            elif path[-1] == 'd':
                new_boxes[box_index] = (new_boxes[box_index][0] + 1, new_boxes[box_index][1])
            elif path[-1] == 'l':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] - 1)
            elif path[-1] == 'r':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] + 1)

            #new_states.append((path, SokobanState(player=new_player, boxes=new_boxes), len(path)))  # Considering each move's cost
            new_states.append((path, SokobanState(player=new_player, boxes=new_boxes), 4))  # Same cost for all moves
        return new_states


# Acknowledgement of taking help of online sources.

class Heuristic:

    def __init__(self, problem):
        self.problem = problem
        self.buffer = self.calc_cost()
        self.box_state = self.problem.init_boxes
        self.memo = dict()

    def calc_manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def calc_euclid(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]))

    # def heuristic1(self, s):
    #     box_pos = s.data[1:]
    #     targets = self.problem.targets
    #     targets_left = len(targets)
    #     total = 0
    #     for ind, box in enumerate(box_pos):
    #         total += self.calc_manhattan(box, targets[ind])
    #         if box in targets:
    #             targets_left -= 1
    #     return total * targets_left

    def calc_cost(self):

        def flood(x, y, cost):
            if not visited[x][y]:

                # Update cost if less than previous target
                if buffer[x][y] > cost:
                    buffer[x][y] = cost
                visited[x][y] = True

                # Check adjacent floors
                if self.problem.map[x - 1][y].floor:
                    flood(x - 1, y, cost + 1)
                if self.problem.map[x + 1][y].floor:
                    flood(x + 1, y, cost + 1)
                if self.problem.map[x][y - 1].floor:
                    flood(x, y - 1, cost + 1)
                if self.problem.map[x][y + 1].floor:
                    flood(x, y + 1, cost + 1)

        buffer = [[infinity for _ in j] for j in self.problem.map]
        #for target in self.problem.init_boxes:
        for target in self.problem.targets:
            visited = [[False for _ in i] for i in self.problem.map]
            flood(target[0], target[1], 0)

        return buffer

    def box_moved(self, current):
        count = 0
        for ind, val in enumerate(current):
            if val != self.box_state[ind]:
                count += 1
        self.box_state = current
        return count

    def heuristic2(self, s):
        box_pos = s.data[1:]
        if box_pos in self.memo:
            return self.memo[box_pos]
        targets = self.problem.targets
        matrix = self.problem.map
        box_moves = self.box_moved(box_pos)
        total = 0

        targets_dist = len(targets)
        for ind, box in enumerate(box_pos):
            total += self.calc_manhattan(box, targets[ind])
            if box in targets:
                targets_dist -= 1

        targets_left = len(targets)

        for val in box_pos:
            if val not in targets:
                if matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = infinity
                    return infinity
                elif matrix[val[0] - 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = infinity
                    return infinity
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] - 1].wall:
                    self.memo[box_pos] = infinity
                    return infinity
                elif matrix[val[0] + 1][val[1]].wall and matrix[val[0]][val[1] + 1].wall:
                    self.memo[box_pos] = infinity
                    return infinity

            else:
                targets_left -= 1
            
            total += self.buffer[val[0]][val[1]]
        self.memo[box_pos] = total * box_moves * (targets_left + targets_dist)
        return total * box_moves * (targets_left+targets_dist)
        #return total *5* box_moves *3* targets_left


# solve sokoban map using specified algorithm
def solve_sokoban(map, dead_detection,file):
    # problem algorithm
    problem = SokobanProblem(map, dead_detection)

    # search algorithm
    search = AStar(Heuristic(problem).heuristic2)

    y = ''
    z = ''
    # solve problem
    search.solve(problem)

    print()
    box_moves = []

    file_name = 'solution'
    
    for i in range(0,len(search.actions)):
        x = search.actions[i] + ''
        x = x[:-1] + x[-1].upper()
        #y += search.actions[i]
        y += x
        box_moves.append(x[-1])
        z += box_moves[i]


    for i in range(0,len(box_moves)):
        z += box_moves[i] + ''
    file_obj = open(f'{file_name}.txt', 'a+')
    file_obj.write(y)
    file_obj.write("\n")
    file_obj.close()
    print('Player moves are: ', y)
    
    print("Box Moves: ", z)
    print('Total Box Moves: {}'.format(len(search.actions)))
    return search.totalCost, search.actions, search.numStatesExplored
    

def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip()[:1] == ';':
                if start: break
                if line.strip() == ';' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else:
                    break
    if not found:
        raise Exception('The Level ' + level + ' is not found')
    return map.strip('\n')


# extract all levels from file
def get_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip()[:1] == ';':
                levels += [line.strip()[1:]]
    return levels


# read level map from file, returns map represented as string
def solve_map(file, level,  dead):
    map = read_map_from_file(file, level)
    print(map)
    start_time = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, dead,file)
    stop_time = datetime.datetime.now()
    print('Time taken: {:.3f} seconds and total states explored: {}'.format(
        (stop_time - start_time).seconds + (stop_time - start_time).microseconds / 1e6, nstates))
    # print(sol)
    seq = ''.join(sol)
    print("Total player moves: ",len(seq))

 

def main():

    algorithm = 'fa2'
    level = 'all'
    #file = 'projectTest.xsb'
    #file = 'levels - Copy.txt'
    #file = 'levels.xsb'
    #file = 'hehe.txt'
    file = 'testExamples.xsb'
    dead = True
    start_time = datetime.datetime.now()
    if os.path.exists("solution.txt"):
        os.remove("solution.txt")
    if level == 'all':
        levels = get_levels(file)
        for level in levels:
            print('Starting level {}'.format(level))
            solve_map(file, level, dead)

    stop_time = datetime.datetime.now()
    print('\n\nTotal Time taken: {:.3f} seconds'.format((stop_time - start_time).seconds + (stop_time - start_time).microseconds/1e6))

if __name__ == '__main__':
    main()
