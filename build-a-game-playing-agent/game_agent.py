"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!

    # heuristic #1:
    
    """The heuristic assumes that, for active player, it will enjoy certain 
    advantage when its legal moves overlap with those of its opponent, because 
    this situation means the active player can choose to move to a location that 
    is in the list of opponent’s potential moves, and therefore diminish opponent’s 
    choices. So I award an extra 1 point to the active player. Other than that, the 
    heuristic is the same as “improved” one."""

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    
    #check for interscection of both player's moves
    moves_inter = set(own_moves).intersection(opp_moves)
    
    if player == game.active_player:
        return float(len(own_moves) - len(opp_moves) + bool(moves_inter))
    else:
        return float(len(own_moves) - len(opp_moves) - bool(moves_inter))
    

    # heuristic #2:
    
    """This heuristic assumes that, it is better stay in the center of the game 
    board. Put another way, need to stay away from boarder and corner. This is 
    because when staying in the center, there are potentially more choices to 
    strategically move to a preferable area. For example, if there are a lot of 
    blank spaces on the right hand side of the board, the player can relatively 
    easily move to that side. For comparison, if the player is at the boarder or 
    corner, it will be harder to get out and move to a preferable side. Other than 
    that, the heuristic is the same as the “Improved” one."""

    # function punish border and corner locations
    def value_centerism(move):
        value = 2
        row, col = move
        if row < 2 or row > game.height - 3:
            value -= 1
            if row < 1 or row > game.height - 2:
                value -= 1
        if col < 2 or col > game.width - 3:
            value -= 1
            if col < 1 or col > game.width - 2:
                value -= 1
        return value

    aplayer = game.active_player
    moves_a = game.get_legal_moves(aplayer)

    if moves_a:
        
        iplayer = game.inactive_player
        moves_i = game.get_legal_moves(iplayer)
        val_a = len(moves_a) + value_centerism(game.get_player_location(aplayer))
        val_i = len(moves_i) + value_centerism(game.get_player_location(iplayer))

        return float(val_a - val_i) if player == aplayer else float(val_i - val_a)
    else:
        return float("-inf") if player == game.active_player else float("inf")

    # heuristic #3

    """This heuristic is based on “open” version. It adds a further check for a 
    certain scenario where the inactive player only has one legal move, and that 
    one can be immediately occupied by the active player. If this is the case, the 
    active player can choose this step and win. Put another way, this heuristic is 
    looking for a certain scenario where the actively player can win in the next 
    ply. It allows for a quick “peek” at one more ply so as to expand the plies."""

    aplayer = game.active_player
    moves_a = game.get_legal_moves(aplayer)

    if moves_a:

        moves_i = game.get_legal_moves(game.inactive_player)
        moves_inter = set(moves_a).intersection(moves_i)
        
        # check for immediate win in next ply
        if len(moves_inter) == 1 and list(moves_inter) == moves_i:
            return float("-inf") if player == game.active_player else float("inf")
        
        return float(len(moves_a)) if player == aplayer else float(len(moves_i))
    else:
        return float("-inf") if player == game.active_player else float("inf")

#    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        if not len(legal_moves):
            return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            move = None
            if not self.iterative:
                if self.method == 'minimax':
                    score, move = self.minimax(game, self.search_depth, maximizing_player=True)
                else:
                    score, move = self.alphabeta(game, self.search_depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
            else:
                depth_iter = 0
                while True:
                    if self.method == 'minimax':
                        score, move = self.minimax(game, depth_iter, maximizing_player=True)
                    else:
                        score, move = self.alphabeta(game, depth_iter, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
                    depth_iter += 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # if no moves can win, still choose a move instead of giving up immediately
        return move if move else legal_moves[0]

#        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        
        # identify who is the original player
        original_player = game.active_player if maximizing_player else game.inactive_player
        moves = game.get_legal_moves(game.active_player)
        
        # if checkmate of reaching the last ply, return score
        if not (depth and moves):
           
            return (self.score(game, original_player), 
                    moves[0] if moves else (-1, -1))
        else:
            # recursive search
            list_scores_moves = [(self.minimax(game.forecast_move(move), depth-1, not maximizing_player)[0], move) for move in moves]
            return (max(list_scores_moves) if maximizing_player else min(list_scores_moves))

#        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!


        # identify who is the original player
        original_player = game.active_player if maximizing_player else game.inactive_player
        moves = game.get_legal_moves(game.active_player)
        
        # if checkmate of reaching the last ply, return score
        if not (depth and moves):
            return (self.score(game, original_player), 
                    moves[0] if moves else (-1, -1))
        else:
            # recursive search
            scores = []
            if maximizing_player:
                
                for move in moves:
                    score, _ = self.alphabeta(game.forecast_move(move), 
                                           depth-1, 
                                           alpha, 
                                           beta, 
                                           not maximizing_player)
                    scores.append(score)
                    if score >= beta:
                        return (score, move)
                    alpha = max(alpha, score)
                return max(list(zip(scores, moves)))
            else:
                for move in moves:
                    score, _ = self.alphabeta(game.forecast_move(move), 
                                           depth-1, 
                                           alpha, 
                                           beta, 
                                           not maximizing_player)
                    scores.append(score)
                    if score <= alpha:
                        return (score, move)
                    beta = min(beta, score)
                return min(list(zip(scores, moves)))

#        raise NotImplementedError
