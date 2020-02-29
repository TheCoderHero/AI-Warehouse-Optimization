# Optimizing Warehouse Flow - Q-Learning

#Importing the libraries
import numpy as np

#Setting the paramters gamma and alpha for the Q-Learning "Bellman Equation"
gamma = 0.75
alpha = 0.9

# PART 1 - DEFINING THE ENVIRONMENT

# DEFINE STATES

# CREATE MAPPING OF LOCATIONS TO STATES
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# REVERSE MAPPING OF STATES TO LOCATIONS
state_to_location = {state: location for location, state in location_to_state.items()}

# DEFINE ACTIONS
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# DEFINE REWARDS
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,0,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# INITIALIZE THE Q-VALUES
Q = np.array(np.zeros([12, 12]))

# IMPLEMENT THE Q-LEARNING PROCESS

# CREATE FUNCTION TO RUN Q-LEARNING BELLMAN EQUATION
def qLearningProcess (R, iterations):
    # Run the process 1000 times
    for learningTests in range(iterations):
        
        # Select a random state to start in
        current_state = np.random.randint(0, 12)
        
        # Create a list of playable actions
        playable_actions = []
        
        # Loop through a given states rewards
        for reward in range(12):
            
            # If the reqard is greater than 0
            if R[current_state, reward] > 0:
                
                # Add that reward to the list
                playable_actions.append(reward)
                
        # Choose one of the actions at random
        next_state = np.random.choice(playable_actions)
        
        # Compute the Temporal Difference
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        
        # Update the Q-Value based on the TD calculation
        Q[current_state, next_state] += alpha * TD

# CREATE FUNCTION FOR OPTIMIZED ROUTE
def route(starting_location, ending_location):
    
    # Create a copy of the R array
    R_new = np.copy(R)
    
    # Get ending state from location dictionary
    ending_state = location_to_state[ending_location]
    
    # Update ending_state R Value
    R_new[ending_state, ending_state] = 1000
    
    # Perform the Optimization Function
    qLearningProcess(R_new, 1000)
    
    # Initialize list with the starting location
    route = [starting_location]
    
    # Create variable to hold the next location
    next_location = starting_location
    
    # Loop through the locations appending each location to the list
    while (next_location != ending_location):
        
        # Create variable to hold starting state
        starting_state = location_to_state[starting_location]
        
        # Get highest Q-Value for the starting state
        next_state = np.argmax(Q[starting_state,])
        
        # Create variable to hold letter of next state
        next_location = state_to_location[next_state]
        
        # Add the location to the list of locations
        route.append(next_location)
        
        # Update starting location to handle while loop
        starting_location = next_location
        
    return route

# PART 3 - GOING INTO PRODUCTION

# PRINT FINAL ROUTE OF AI WAREHOUSE
print('Route: ')
route('E', 'G')