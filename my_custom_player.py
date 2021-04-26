
from sample_players import DataPlayer

class CustomPlayer(DataPlayer):
    
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
#         self.queue.put(random.choice(state.actions()))

    
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.minimax(state, depth=3))
#             self.queue.put(self.alpha_beta_search(state, depth=3))

    def minimax(self, state, depth):
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

 
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
#         return len(own_liberties) - len(opp_liberties)
    
        #Calculate the midpoint of open spaces 
        from isolation import DebugState
        bitstate = DebugState.from_state(state).bitboard_string
        bitstate ='0'*(13*9-len(bitstate)-1) +bitstate +'0'
        bitstate = bitstate[::-1]
        open_space_midpoint=[0,0]
        count=0
        for cols in range(9):
            for rows in range(13):
                pos  = (cols)*13+rows

                v = int(bitstate[pos])
                open_space_midpoint[0] += v*(rows-1)
                open_space_midpoint[1] += v*(cols)
                count +=v
                
        open_space_midpoint[0] /=count
        open_space_midpoint[1] /=count
        
        # Calculate the distances to the midpoint
        own_loc = list(DebugState.ind2xy(own_loc))
        opp_loc = list(DebugState.ind2xy(opp_loc))
        
        own_loc[0] = abs(own_loc[0]-open_space_midpoint[0])**2
        own_loc[1] = abs(own_loc[1]-open_space_midpoint[1])**2
        
        opp_loc[0] = (opp_loc[0]-open_space_midpoint[0])**2
        opp_loc[1] = (opp_loc[1]-open_space_midpoint[1])**2
        
        own_dist = (own_loc[0]+own_loc[1])**0.5
        opp_dist = (opp_loc[0]+opp_loc[1])**0.5
    
        # Returns: a weighted score based on #own moves and distance to 
        # open spaces midpoint
        # Maximises: relative #own moves and opponent's distance relative to own
        # Weight Schedule: weight distance measure in line with spaces available
        # Why?:
        # - #moves discussed in class
        # - The relative distance to open space midpoint- try to keep your opponent 
        # as far away from the general area of open space
        # - Spaces become more disperse over time and the central
        # point loses its significance relative to #moves
        # Why
        if len(own_liberties) ==0:return len(own_liberties)-len(opp_liberties)
#         ret = (len(own_liberties) - len(opp_liberties))*(99*11-count)/99/11 + (opp_dist-own_dist)*(count)/99/11
        ret = (len(own_liberties) - len(opp_liberties))*(99*11-count) + (opp_dist-own_dist)*(count)
    
#         ret = (len(own_liberties) - len(opp_liberties)) 
        #print('ret',ret)
        return ret
#         return  opp_dist-own_dist
    
    
