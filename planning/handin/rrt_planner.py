"""
Assignment #2 Template file
"""
import random
import math
from xxlimited import new
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for the problem setup given by the RRT_DUBINS_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_planner.py. Your implementation
   can be tested by running RRT_DUBINS_PROBLEM.PY (check the main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class to ease implementation.
3. Your solution must meet all the conditions specificed below.
4. Below are some do's and don'ts for this problem as well.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random iterations
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the node class to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while the parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area.

DO(s) and DONT(s)
-------------------
1. Do not rename the file rrt_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""
def get_nearest(node_list, rnd):
   """
   input: node list, random node
   output: index of the node in the node list at the minimum distance with
            the random node
   """
   dist = []
   dist = [(node.x - rnd.x)**2 + (node.y - rnd.y)**2 for node in node_list]
   min_ind = dist.index(min(dist))
   
   return min_ind

def path_to_goal(rrt_dubins):
   """
   input: node list
   output: list of all the nodes leading to the goal
   """
   node = rrt_dubins.node_list[-1]
   path = [node]
   total_cost = 0
   while(node.parent != None):
      try:
         cost = rrt_dubins.calc_new_cost(node, node.parent)
      except:
         break
      path.append(node.parent)
      node = node.parent
      total_cost += cost
      
   path.reverse()
   return path


def rrt_planner(rrt_dubins, display_map=False):
   """
      Execute RRT planning using Dubins-style paths. Make sure to populate the node_list.

      Inputs
      -------------
      rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                     problem specification
      display_map - (boolean) flag for animation on or off (OPTIONAL)

      Outputs
      --------------
      (list of nodes) This must be a valid list of connected nodes that form
                     a path from start to goal node

      NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
      to populate rrt_dubins.nodes_list with all valid RRT nodes.
   """
   # set a seed for reproducibility
   # np.random.seed(101)
   # LOOP for max iterations
   i = 0
   min_goal_th = 1
   bias = 0.1
   while i < rrt_dubins.max_iter:
      i += 1

      # Generate a random vehicle state (x, y, yaw)
      if (np.random.random() < 1 - bias):
         v_x = np.random.randint(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1])
         v_y = np.random.randint(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1])
         v_yaw = np.random.uniform(-np.pi, np.pi)
         rnd_node = rrt_dubins.Node(v_x, v_y, v_yaw)
      else:
         rnd_node = rrt_dubins.goal
      # Find an existing node nearest to the random vehicle state
      
      nearest_ind = get_nearest(rrt_dubins.node_list, rnd_node)
      nearest_node = rrt_dubins.node_list[nearest_ind]
   
      new_node = rrt_dubins.propogate(nearest_node, rnd_node) #example of usage
      
      # Check if the path between nearest node and random state has obstacle collision
      # Add the node to nodes_list if it is valid
      if rrt_dubins.check_collision(new_node):
         rrt_dubins.node_list.append(new_node) # Storing all valid nodes
         
      # Draw current view of the map
      # PRESS ESCAPE TO EXIT
      if display_map:
         rrt_dubins.draw_graph()


      # Check if new_node is close to goal
      if (rrt_dubins.node_list[-1].is_state_identical(rrt_dubins.goal)):
         # print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
         return path_to_goal(rrt_dubins) 
         

      if i == rrt_dubins.max_iter:
         print('reached max iterations')
         return None


   # Return path, which is a list of nodes leading to the goal...
   return None
