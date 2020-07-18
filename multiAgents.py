# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # No need to add the code here

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Firstly find the the game state score
        game_State_score = successorGameState.getScore()

        # Assigning initial pacman to ghost location
        pacman_ghost_loc = 1

        # Assigning the closest threshold between ghost and pacman
        closest_pacman_ghost = 0

        # pacman position:
        new_pacman_position = successorGameState.getPacmanPosition()

        # zero val_one_val
        zero_val = 0
        one_val = 1

        # Firstly find all the food locations from the mesh and put it in list
        all_food_points = newFood.asList()

        # as mentioned in the problem considering reciprocal of food to pacman distance
        # Looping over food points
        for food_locations in all_food_points:
            # Calculate manhattan distance between pacman position and food location from list
            # We are using manhattan distance as we are oving in mesh
            food_manhattan_distance = util.manhattanDistance(food_locations, new_pacman_position)
            #checking is distance is zero  or not
            if (food_manhattan_distance) != zero_val:
                # Calculate reciprocal for manhatten distance
                rec_food_manhattan_distance = (1.0 / food_manhattan_distance)

                # As mentioned in problem notes taking reciprocal and adding value with test score
                game_State_score = game_State_score + rec_food_manhattan_distance

        # Calculate all the ghost locations
        ghost_positions = successorGameState.getGhostPositions()

        # as mentioned in the problem considering reciprocal of food to pacman distance
        # looping over the ghost locations
        for all_ghost_points in newGhostStates:

            #calculating the ghost positions
            ghost_pac_distance = all_ghost_points.getPosition()

            # Calculate manhattan distance between pacman position and ghost  location from list
            # We are using manhattan distance as we are moving in mesh
            ghost_manhatten_dist = util.manhattanDistance(ghost_pac_distance, new_pacman_position)

            #Taking difference and reciprocal values
            new_pos_ghost_pac_0 = newPos[zero_val] - ghost_pac_distance[zero_val]

            new_pos_ghost_pac_zero = abs(new_pos_ghost_pac_0)

            #Taking difference and reciprocal values
            new_pos_ghost_pac_1 = abs(newPos[one_val] - ghost_pac_distance[one_val])

            new_pos_ghost_pac_one = abs(new_pos_ghost_pac_1)

            if (new_pos_ghost_pac_zero + new_pos_ghost_pac_one) > one_val:

                # As mentioned in the notes take reciprocal of distance between ghost and pacman
                rec_ghost_manhatten_dist = (1.0 / ghost_manhatten_dist)

                # Adding reciprocal of manhatten distance
                game_State_score = game_State_score + rec_ghost_manhatten_dist
        #returning the final state score for simple reflex agent
        return game_State_score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    # No need to make the changes in this class.
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # We need depth, pacman positions and and agent count to write the minimax function
        # Pacman positions:
        current_pacman_positions = gameState
        # The total number of agents:
        total_game_agents = current_pacman_positions.getNumAgents()

        # calculate the depth
        max_depth_minimax = self.depth * total_game_agents

        # Adding flag
        flag = False

        # calling our get action call where data get processed and minimax get called
        self.call_getaction_minimax(current_pacman_positions, total_game_agents, max_depth_minimax, flag )


        #return the output for get action
        return self.action1

        # write the minimax recursive function
    def call_getaction_minimax(self,  current_pacman_positions, total_game_agents, max_depth_minimax, flag):

        #We can modify the input data here as per requirement

        #Calling the minimax function
        self.minimax_calculator(current_pacman_positions, total_game_agents, max_depth_minimax, flag )




    #Minimax function with following inputs: position, depth, agets and flag
    def minimax_calculator(self, current_pacman_positions, total_game_agents, max_depth_minimax, flag):

        # create the list of max value nodes
        max_value_nodes = list()

        # adding variables to make code more readable
        zero_value = 0
        one_value = 1

        #input flg
        input_flag = flag

        # create the list of min value nodes
        min_value_nodes = list()

        # check whether given node is winning node or node where our agent loose
        pacman_winning_node = current_pacman_positions.isWin()
        pacman_loosing_node = current_pacman_positions.isLose()

        if (pacman_winning_node == True or pacman_loosing_node == True):
            return self.evaluationFunction(current_pacman_positions)

        # check the output when depth is more than 0

        if max_depth_minimax > zero_value:

            # finding the depth
            act_depth = max_depth_minimax % total_game_agents
            # Agents if depth is zero
            if act_depth == zero_value:

                calculated_agents = zero_value

            else:
                calculated_agents = total_game_agents - act_depth

                # As per given data legal actions method provide us the list of posible actions pacman can take
            possible_pac_positions = current_pacman_positions.getLegalActions(calculated_agents)

            for pacman_action_node in possible_pac_positions:

                # find the next pacman position
                possible_next_position = current_pacman_positions.generateSuccessor(calculated_agents,
                                                                                    pacman_action_node)

                # Check calculated pacman locations position
                if calculated_agents == zero_value:

                    # calling the minimax function recursively
                    t_g_a = total_game_agents
                    depth_update = max_depth_minimax - 1
                    p_a_n_i = pacman_action_node
                    max_value_nodes.append(
                        (self.minimax_calculator(possible_next_position, t_g_a, depth_update, input_flag), p_a_n_i, ))

                    # Calculate the max value from the max_value_nodes list
                    max_possible_output = max(max_value_nodes)

                    # Assign this value to value_max
                    self.value_max = max_possible_output[zero_value]

                    self.action1 = max_possible_output[one_value]
                # Checking for the ghost locations and ghosts minimizing function
                else:
                    # In minimax ghost always try to minimize pacmans function
                    # Adding minimum value in min_value_nodes list
                    t_g_a_e = total_game_agents
                    depth_update_el = max_depth_minimax - 1
                    p_a_n = pacman_action_node
                    p_n_p = possible_next_position

                    min_cal = (self.minimax_calculator(p_n_p, t_g_a_e, depth_update_el, input_flag ), p_a_n)
                    #appending value in list
                    min_value_nodes.append(min_cal)

                    # Find min value which can achived by ghost
                    min_possible_ghost_value = min(min_value_nodes)

                    # Add this value in value_min
                    self.value_min = min_possible_ghost_value[zero_value]

            # making function dynamic to operate
            zero_val = 0
            if calculated_agents == zero_val:
                # This value will return max output from max_value_nodes
                return self.value_max
            else:
                # This value will return min output from min_value_nodes
                return self.value_min

        else:
            return self.evaluationFunction(current_pacman_positions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # we will create 2 separate function for alpha and beta values
        #Zero val
        zero_val = 0
        depth_alpha_beta = self.depth

        # fetching the mesh
        current_pacman_positions = gameState

        #set flag
        flag = False

        # min infinite
        min_inf = float('-inf')

        # maximum infinite
        max_inf = float('+inf')

        # for alpha beta pruning we have to keep initial values as -infity to +infynity
        # calling maximizing function first

        #We are calling alpha function and this function will call beta one
        cal_data, alpha_beta_out = self.getAction_call_alpha_beta(current_pacman_positions, zero_val, depth_alpha_beta, min_inf, max_inf, flag)

        #return the output for get action

        return alpha_beta_out

    def getAction_call_alpha_beta(self,  current_pacman_positions,  total_game_agents,
                              max_depth_alpha_beta,alpha_points, beta_points, flag):

        #We can modify the input data here as per requirement

        # We are calling alpha function and this function will call beta one
        data, value = self.alpha_pacman_function(current_pacman_positions, total_game_agents, max_depth_alpha_beta,
                                                              alpha_points, beta_points, flag)
        return data, value

    def alpha_pacman_function(self, current_pacman_positions,  total_game_agents,
                              max_depth_alpha_beta,alpha_points, beta_points, flag):

        # declare the negative infinity
        neg_infinity = float('-inf')

        # assigning changing value
        alpha_change_inf = neg_infinity

        #set flag
        input_flag =  flag

        # check whether given node is winning node
        pacman_winning_node = current_pacman_positions.isWin()

        # loosing node
        pacman_loosing_node = current_pacman_positions.isLose()

        # To make code more readable
        zero_value = 0
        one_value = 1

        if (pacman_winning_node == True or pacman_loosing_node == True):
            # checking whether its winning or loosing node
            return self.evaluationFunction(current_pacman_positions), 'none'

        # As per given data legal actions method provide us the list of posible actions pacman can take
        possible_pac_positions = current_pacman_positions.getLegalActions(total_game_agents)

        # finding the best possible position
        optimal_position_pacman = possible_pac_positions[zero_value]

        # Loop over the all tha calculated legal actions

        for possible_positions in possible_pac_positions:
            # add previous infinity
            before_infinite = alpha_change_inf

            # find the next pacman position
            possible_next_position = current_pacman_positions.generateSuccessor(total_game_agents, possible_positions)

            # check whether given node is winning node
            pacman_winning_node_succ = possible_next_position.isWin()

            # loosing node
            pacman_loosing_node_succ = possible_next_position.isLose()

            # for leaf nodes or if the game finishes
            if pacman_winning_node_succ == True or pacman_winning_node_succ == True or pacman_loosing_node_succ == True:

                #calculate evaluation function value
                eval_function = self.evaluationFunction(possible_next_position)

                #final alpha change value
                alpha_change_inf = max(alpha_change_inf, eval_function)
            else:
                # Call the function beta_ghost_pruning
                beta = beta_points

                incre_game_a = total_game_agents + 1

                pnp = possible_next_position

                depth_ab = max_depth_alpha_beta

                a_c_i = alpha_change_inf
                # calculate bete function output
                beta_output = self.beta_ghost_pruning(pnp, incre_game_a, depth_ab, alpha_points, beta, input_flag)
                alpha_change_inf = max(a_c_i,beta_output )

            # Check the pruning conditions
            # skip the nodes which no required
            if alpha_change_inf > beta_points:
                return alpha_change_inf, possible_positions

            # selecting maximum value from tuple
            alpha_points = max(alpha_points, alpha_change_inf)

            #If value are not equel
            if alpha_change_inf != before_infinite:
                #assign value  to optimal_position_pacman
                optimal_position_pacman = possible_positions

        return alpha_change_inf, optimal_position_pacman

    def beta_ghost_pruning(self, current_pacman_positions,  total_game_agents,
                           max_depth_alpha_beta,alpha_points, beta_points, flag):

        # Set the flag
        change_depth_flag = False

        # firstly calculate positive infinite value
        pos_infinity = float('inf')

        # firstly declare beta changing variable
        beta_change_inf = pos_infinity

        #input flag
        input_flag = flag

        # check whether given node is winning node
        pacman_winning_node = current_pacman_positions.isWin()

        # loosing node
        pacman_loosing_node = current_pacman_positions.isLose()

        # To make code more readable
        value_zero = 0
        value_one = 1

        # checking whether current node is of winning or loosing node
        if pacman_winning_node == True or pacman_loosing_node == True:
            #calculate eval function output
            evel_out = self.evaluationFunction(current_pacman_positions)
            #if the node is winning point node or loosing point node
            return evel_out, 'none'

        # As per given data legal actions method provide us the list of posible actions pacman can take
        possible_pac_positions = current_pacman_positions.getLegalActions(total_game_agents)

        # looping over all possible legal actions
        for possible_position_beta in possible_pac_positions:

            # find the next pacman position
            possible_next_position = current_pacman_positions.generateSuccessor(total_game_agents,
                                                                                possible_position_beta)

            # checking the win loose and depth condition
            # check whether given node is winning node
            pacman_winning_node_succ = possible_next_position.isWin()

            # loosing node
            pacman_loosing_node_succ = possible_next_position.isLose()

            #calculate the agent values
            cal_agents = current_pacman_positions.getNumAgents()

            #checking winning loosing and zero value condition
            if pacman_winning_node_succ == True or max_depth_alpha_beta == value_zero or pacman_loosing_node_succ == True:
                # find evaluation function value
                eval_function_value = self.evaluationFunction(possible_next_position)
                #Calculate beta
                beta_change_inf = min(beta_change_inf, eval_function_value)

            elif total_game_agents == (cal_agents - value_one):
                # check the depth flag
                if change_depth_flag == False:
                    max_depth_alpha_beta = max_depth_alpha_beta - value_one
                    # set the visited level
                    change_depth_flag = True

                if max_depth_alpha_beta == value_zero:
                    #find eval function value
                    eval_function = self.evaluationFunction(possible_next_position)
                    # assigning value to the variable
                    beta_change_inf = min(beta_change_inf, eval_function)
                else:
                    p_n_p = possible_next_position

                    alpha = alpha_points

                    beta = beta_points

                    d = max_depth_alpha_beta
                    #find first value from alpha function output
                    alpha_first_val = self.alpha_pacman_function(p_n_p, 0, d, alpha, beta, input_flag)[0]
                    # call the function alpha_pacman_function
                    beta_change_inf = min(beta_change_inf, alpha_first_val)

            else:
                p_n_p = possible_next_position

                alpha = alpha_points

                beta = beta_points

                d = max_depth_alpha_beta

                t_g_a = total_game_agents + 1
                #calculate beta function value
                beta_val = self.beta_ghost_pruning(p_n_p, t_g_a, d, alpha, beta, input_flag)
                # call the function alpha_pacman_function
                beta_change_inf = min(beta_change_inf, beta_val)
            # Checking for the minima

            if beta_change_inf < alpha_points:
                # returning calculated beta values
                return beta_change_inf
            # assigning minimum beta values
            beta_points = min(beta_points, beta_change_inf)
        # returning minimum value here
        return beta_change_inf


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # fetching the mesh
        current_pacman_positions = gameState

        # firstly find the agents
        total_calculated_agents = current_pacman_positions.getNumAgents()

        # calculate the maximum depth
        max_depth_equimax = self.depth * total_calculated_agents

        #assign with the flag
        flag = False

        # calling the equimax calculator function
        ##
        self.getActionCallEqu(current_pacman_positions, total_calculated_agents, max_depth_equimax, flag)

        return self.action1

    def getActionCallEqu(self,current_pacman_positions,  total_game_agents, max_depth_equimax_in, flag ):

        #Write code here if you want to manipulate the given data

        #calling the newly defined equimax funation
        self.equimax_calculator(current_pacman_positions, total_game_agents, max_depth_equimax_in, flag)


    #calling newly defined equimax_calculator function
    def equimax_calculator(self, current_pacman_positions,  total_game_agents, max_depth_equimax_in, flag):

        # create the list of max value nodes
        max_value_nodes = list()

        # zero val to make the programe dynamic
        zero_val = 0
        one_val = 1

        # create the list of min value nodes
        possible_value_nodes = list()

        # check whether given node is winning node
        pacman_winning_node = current_pacman_positions.isWin()

        #assign flag
        input_flag = flag

        # check whether given node is loosing node
        pacman_loosing_node = current_pacman_positions.isLose()

        if pacman_winning_node == True or pacman_loosing_node == True:
            #find eval value output
            eval_function = self.evaluationFunction(current_pacman_positions)
            return eval_function

        if max_depth_equimax_in > zero_val:
            # calculate the relation in depth and number of agents
            depth_agent_reminder = max_depth_equimax_in % total_game_agents
            # checking whether depth_agent_reminder is zero or not
            if depth_agent_reminder == zero_val:
                pac_ghost_count = zero_val

            else:
                # if depth_agent_reminder is not zero
                pac_ghost_count = total_game_agents - (depth_agent_reminder)

            # As per given data legal actions method provide us the list of posible actions pacman can take
            possible_pac_positions = current_pacman_positions.getLegalActions(pac_ghost_count)

            # looping over all possible legal pacman actions

            for possible_position_equival in possible_pac_positions:
                # set variable to fetch data from possible_value_nodes list
                values_nodes_in = 0.0

                # find the next pacman position
                possible_next_position = current_pacman_positions.generateSuccessor(pac_ghost_count,
                                                                                    possible_position_equival)

                # condition when agents are zero
                if pac_ghost_count == zero_val:
                    # recursively calling the function
                    p_n_p = possible_next_position
                    next_depth = max_depth_equimax_in - one_val
                    t_g_a = total_game_agents
                    p_p_e = possible_position_equival
                    #calculate equi_max output value
                    equimax_out = (self.equimax_calculator(p_n_p,  t_g_a, next_depth, input_flag), p_p_e)

                    #Appending value to node
                    max_value_nodes.append(equimax_out)

                    # As it is equimax function, we try to find max val
                    # fetching max value from max-value_node list
                    optimum_max_val = max(max_value_nodes)

                    # set the value to action and value_max
                    self.value_max = optimum_max_val[zero_val]

                    self.action1 = optimum_max_val[one_val]

                else:
                    # if agents are not zero call the function recursively
                    p_n_p = possible_next_position

                    new_depth = max_depth_equimax_in - one_val

                    t_g_a = total_game_agents

                    #Calculate output for equimax
                    equ_final_output = (self.equimax_calculator(p_n_p,
                                                                         t_g_a, new_depth, input_flag),
                                                 possible_position_equival)
                    #Appending to the list
                    possible_value_nodes.append(equ_final_output)

                    #find the size of possible values list
                    size_possible_value_nodes = len(possible_value_nodes)

                    # check value from list possible_value_nodes
                    for elements_possible_value_nodes in possible_value_nodes:
                        #Checking for all the possible values
                        values_nodes_in = values_nodes_in + possible_value_nodes[
                            possible_value_nodes.index(elements_possible_value_nodes)][zero_val]
                    #dividing by list length
                    values_nodes_in = values_nodes_in / size_possible_value_nodes

                    self.value_avg = values_nodes_in

            if pac_ghost_count == zero_val:
                #returning the maximum value obtained
                return self.value_max
            else:
                #returning the average value obtained
                return self.value_avg

        else:
            #returning evaluate function
            return self.evaluationFunction(current_pacman_positions)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #To make code redable

    zero_val = 0
    one_val = 1
    # Assign one constant
    closest_food_pac_val = -1
    # assign min distance between gost and pacman
    closest_ghost_location = 1

    # calculate ghost collide probability
    ghost_pac_min = 0

    # better evaluation function will obtain if we take inverse of food and ghost positions
    current_pacman_mesh = currentGameState

    # fetch the food location from given mesh
    pac_food_points = current_pacman_mesh.getFood()

    # fetch the list of food points
    pac_food_point_list = pac_food_points.asList()

    # Now we have to check the pacmans current position:
    pacman_location = current_pacman_mesh.getPacmanPosition()

    # find the ghost states
    ghost_states = current_pacman_mesh.getGhostPositions()

    # calculate the game score
    score_attained = current_pacman_mesh.getScore()

    # iterating over all the food point locations
    for food_points in pac_food_point_list:
        # calculate manhatten distance between pacman location and food points
        pac_food_dist = util.manhattanDistance(pacman_location, food_points)

        if closest_food_pac_val >= pac_food_dist or closest_food_pac_val == -1:
            closest_food_pac_val = pac_food_dist

    for pac_ghost_dist in ghost_states:
        # we are calculating the manhatten distance as its mesh structure, pacman can only move along squares
        dist_ghost_pacman = util.manhattanDistance(pacman_location, pac_ghost_dist)

        closest_ghost_location = closest_ghost_location + dist_ghost_pacman

        if dist_ghost_pacman <one_val or dist_ghost_pacman == one_val:
            ghost_pac_min = ghost_pac_min + one_val

    #find the capsules list
    number_caps = current_pacman_mesh.getCapsules()

    # calculating the length of list of capsules
    number_capsules = len(number_caps)

    #Taking reciprocal values as per mentioned in the example
    # as mentioned in question take reciprocal of closest food point value
    closest_food_pac_float = float(closest_food_pac_val)
    rec_closest_food_pac_float = (1 / closest_food_pac_float)

    # as mentioned in question take reciprocal of closest ghost location and pacman point value
    closest_ghost_location_float = float(closest_ghost_location)
    rec_closest_ghost_location = (1 / closest_ghost_location_float)

    # final evaluation
    r_c_f_p_f = rec_closest_food_pac_float

    n_c = number_capsules

    r_c_g_l = rec_closest_ghost_location

    #substracting the reciprocal values
    sub_reciprocals = r_c_f_p_f - (r_c_g_l) -n_c - ghost_pac_min

    #final optimal ecaluation score
    evaluated_score_achived = score_attained + sub_reciprocals

    return evaluated_score_achived


# Abbreviation
better = betterEvaluationFunction

