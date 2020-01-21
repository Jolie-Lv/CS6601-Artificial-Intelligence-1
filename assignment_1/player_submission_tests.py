#!/usr/bin/env python
import traceback
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer, Player
import platform
import random

if platform.system() != 'Windows':
    import resource

from time import time, sleep

def correctOpenEvalFn(yourOpenEvalFn):
    print()
    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            ["Q1", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", "Q2", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "]
        ]
        sample_board.set_state(board_state, True)
        #test = sample_board.get_legal_moves()
        h = yourOpenEvalFn()
        print('OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board, sample_board.get_active_player())))
    except NotImplementedError:
        print('OpenMoveEvalFn Test: Not implemented')
    except:
        print('OpenMoveEvalFn Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()

def beatRandom(yourAgent):

    """Example test you can run
    to make sure your AI does better
    than random."""

    print("")
    try:
        r = RandomPlayer()
        p = yourAgent()
        game = Board(r, p, 7, 7)
        # game = Board(p, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation(time_limit=1000, print_moves=True)
        print("\n", winner, " has won. Reason: ", termination)
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')
    except:
        print('CustomPlayer Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()

def beatMiniMax(yourAgent, minimaxAgent):

    """Example test you can run
    to make sure your AI does better
    than random."""

    print("")
    try:
        r = minimaxAgent()
        p = yourAgent()
        # game = Board(r, p, 7, 7)
        game = Board(p, r, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation(time_limit=1000, print_moves=True)
        print("\n", winner, " has won. Reason: ", termination)
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')
    except:
        print('CustomPlayer Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()

def minimaxTest(yourAgent, minimax_fn):
    """Example test to make sure
    your minimax works, using the
    OpenMoveEvalFunction evaluation function.
    This can be used for debugging your code
    with different model Board states.
    Especially important to check alphabeta
    pruning"""

    # create dummy 5x5 board
    print("Now running the Minimax test.")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 10000

        player = yourAgent() #using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            ["Q1", " ", " ", " ", " ", "X", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["X", " ", " ", " ", " ", " ", " "],
            [" ", " ", "X", "Q2", "X", " ", " "],
            ["X", "X", "X", " ", "X", " ", " "],
            [" ", " ", "X", " ", "X", " ", " "],
            [" ", " ", "X", " ", "X", " ", " "]
        ]
        sample_board.set_state(board_state, True)

        test_pass = True

        expected_depth_scores = [(1, 4), (2, -2), (3, 4), (4, -2), (5, 2)]
        # expected_depth_scores = [(1, 4)]

        for depth, exp_score in expected_depth_scores:
            move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
            # print(move, score)
            if exp_score != score:
                print("Minimax failed for depth: ", depth)
                test_pass = False

        if test_pass:
            player = yourAgent()
            sample_board = Board(player, RandomPlayer())
            # setting up the board as though we've been playing
            board_state = [
                [" ", " ", " ", " ", "X", " ", "X"],
                ["X", "X", "X", " ", "X", "Q2", " "],
                [" ", "X", "X", " ", "X", " ", " "],
                ["X", "X", "X", " ", "X", "X", " "],
                ["X", " ", "Q1", " ", "X", " ", "X"],
                ["X", " ", " ", " ", "X", "X", " "],
                ["X", " ", " ", " ", "X", " ", " "]
            ]
            sample_board.set_state(board_state, True)

            test_pass = True

            expected_depth_scores = [(1, 5), (2, 5), (3, 5), (4, 6), (5, 6)]
            # expected_depth_scores = [(5, 6)]

            for depth, exp_score in expected_depth_scores:
                move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
                if exp_score != score:
                    # print("Minimax failed for depth: ", depth)
                    print("Lev2: Minimax failed for depth: ", depth, " Score, ", score)
                    test_pass = False

        if test_pass:
            print("Minimax Test: Runs Successfully!")

    except NotImplementedError:
        print('Minimax Test: Not implemented')
    except:
        print('Minimax Test: ERROR OCCURRED')
        print(traceback.format_exc())



def alphabetaTest1(yourAgent, minimax_fn):
    """Example test to make sure
    your alphabeta works, using the
    OpenMoveEvalFunction evaluation function.
    This can be used for debugging your code
    with different model Board states.
    Especially important to check alphabeta
    pruning"""

    # create dummy 3x3 board
    print("Now running the AlphaBeta test.")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 10000

        player = yourAgent() #using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer(), width=3, height=3)
        # setting up the board as though we've been playing
        # board_state = [
        #     [" ",  " ",  "Q2"],
        #     [" ",  "Q1", "X"],
        #     ["X",  "X",  "X"]
        # ]
        # board_state = [
        #     ["X",  " ",  "Q2"],
        #     [" ",  " ",  " "],
        #     ["Q1",  "X", " "]
        # ]
        board_state = [
            [" ",  " ",  "X"],
            [" ",  "Q1",  " "],
            ["Q2",  " ", "X"]
        ]
        sample_board.set_state(board_state, True)

        test_pass = True

        expected_depth_scores = [(2, 0)]

        for depth, exp_score in expected_depth_scores:
            move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
            # print(move, score)
            if exp_score != score:
                print(score, exp_score)
                print("AlphaBeta failed for depth: ", depth)
                test_pass = False

        # if test_pass:
        #     player = yourAgent()
        #     sample_board = Board(player, RandomPlayer())
        #     # setting up the board as though we've been playing
        #     board_state = [
        #         ["X",  " ",  "Q2"],
        #         [" ",  " ",  " "],
        #         ["Q1",  "X", " "]
        #     ]
        #     sample_board.set_state(board_state, True)
        #
        #     test_pass = True
        #
        #     expected_depth_scores = [(2,0)]
        #
        #     for depth, exp_score in expected_depth_scores:
        #         move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
        #         if exp_score != score:
        #             # print("Minimax failed for depth: ", depth)
        #             print("Lev2: AlphaBeta failed for depth: ", depth, " Score, ", score)
        #             test_pass = False

        if test_pass:
            print("AlphaBeta Test: Runs Successfully!")

    except NotImplementedError:
        print('AlphaBeta Test: Not implemented')
    except:
        print('AlphaBeta Test: ERROR OCCURRED')
        print(traceback.format_exc())


def alphabetaTest2(yourAgent, minimax_fn):
    """Example test to make sure
    your alphabeta works, using the
    OpenMoveEvalFunction evaluation function.
    This can be used for debugging your code
    with different model Board states.
    Especially important to check alphabeta
    pruning"""

    # create dummy 5x5 board
    print("Now running the AlphaBeta test.")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 10000

        player = yourAgent() #using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            ["Q1", " ", " ", " ", " ", "X", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["X", " ", " ", " ", " ", " ", " "],
            [" ", " ", "X", "Q2", "X", " ", " "],
            ["X", "X", "X", " ", "X", " ", " "],
            [" ", " ", "X", " ", "X", " ", " "],
            [" ", " ", "X", " ", "X", " ", " "]
        ]
        sample_board.set_state(board_state, True)

        test_pass = True

        expected_depth_scores = [(1, 4), (2, -2), (3, 4), (4, -2), (5, 2)]
        # expected_depth_scores = [(3,4)]

        for depth, exp_score in expected_depth_scores:
            move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
            # print(move, score)
            if exp_score != score:
                print(score, exp_score)
                print("AlphaBeta failed for depth: ", depth)
                test_pass = False

        if test_pass:
            player = yourAgent()
            sample_board = Board(player, RandomPlayer())
            # setting up the board as though we've been playing
            board_state = [
                [" ", " ", " ", " ", "X", " ", "X"],
                ["X", "X", "X", " ", "X", "Q2", " "],
                [" ", "X", "X", " ", "X", " ", " "],
                ["X", "X", "X", " ", "X", "X", " "],
                ["X", " ", "Q1", " ", "X", " ", "X"],
                ["X", " ", " ", " ", "X", "X", " "],
                ["X", " ", " ", " ", "X", " ", " "]
            ]
            sample_board.set_state(board_state, True)

            test_pass = True

            expected_depth_scores = [(1, 5), (2, 5), (3, 5), (4, 6), (5, 6)]
            # expected_depth_scores = [(3, 5)]

            for depth, exp_score in expected_depth_scores:
                move, score = minimax_fn(player, sample_board, time_left, depth=depth, my_turn=True)
                if exp_score != score:
                    # print("Minimax failed for depth: ", depth)
                    print("Lev2: AlphaBeta failed for depth: ", depth, " Score, ", score)
                    test_pass = False

        if test_pass:
            print("AlphaBeta Test: Runs Successfully!")

    except NotImplementedError:
        print('AlphaBeta Test: Not implemented')
    except:
        print('AlphaBeta Test: ERROR OCCURRED')
        print(traceback.format_exc())

def random_board(simulate_till=4):

    def time_left():
        return 100000

    randomAgent1 = RandomPlayer()
    randomAgent2 = RandomPlayer()

    game = Board(randomAgent1, randomAgent2)
    move_history = []
    for move_idx in range(simulate_till):
        if move_idx == 0:
            curr_move = (3,3,False)
        elif move_idx == 1: # Non mirrorable moves
            curr_move = random.choice(((1, 2, False), (1, 4, False), (2, 5, False), (4, 5, False), \
                                       (5, 4, False), (5, 2, False), (4, 1, False), (2, 1, False)))
        else:
            curr_move = game.__active_player__.move(game, time_left)
            curr_move = (curr_move[0], curr_move[1], bool(curr_move[2]))

        if curr_move not in game.get_active_moves():
            raise Exception("Illegal move played")

        # Append new move to game history
        if game.__active_player__ == game.__player_1__:
            move_history.append([curr_move])
        else:
            move_history[-1].append(curr_move)

        is_over, winner = game.__apply_move__(curr_move)

        if is_over:
            raise("Game over while simulating board")


    return game, move_history

def test_agent(myAgent, baselineAgent):
    initial_board = None
    p1 = myAgent()
    p2 = baselineAgent()
    games_won = 0
    games_lost = 0
    game_historiesp1, game_historiesp2 = [], []
    for game_no in range(10):
        initial_board, move_history_rand = random_board()
        new_board = Board(p1, p2)
        new_board.set_state(initial_board.get_state(), p1_turn=True)
        new_board
        winner, move_history, reason = new_board.play_isolation(time_limit=1000, print_moves=False)
        move_history = move_history_rand + move_history

        print("Game#%d: %s, %s" %(game_no+1, winner, reason))
        if "CustomPlayer - " not in winner:
            games_lost += 1
            game_historiesp2.append(move_history)
        else:
            games_won += 1
            game_historiesp1.append(move_history)
    print("Win percent: %f" %(games_won/(games_lost+games_won)))
    return game_historiesp1, game_historiesp2

def single_test_agent(myAgent, baselineAgent):
    initial_board = None
    p1 = myAgent()
    p2 = baselineAgent()
    games_won = 0
    games_lost = 0
    game_historiesp1, game_historiesp2 = [], []
    for game_no in range(1):
        initial_board, move_history_rand = random_board(simulate_till=2)
        print()
        new_board = Board(p1, p2)
        new_board.set_state(initial_board.get_state(), p1_turn=True)
        new_board
        winner, move_history, reason = new_board.play_isolation(time_limit=1000, print_moves=True)
        move_history = move_history_rand + move_history

        print("Game#%d: %s, %s" %(game_no+1, winner, reason))
        if "CustomPlayer - " not in winner:
            games_lost += 1
            game_historiesp2.append(move_history)
        else:
            games_won += 1
            game_historiesp1.append(move_history)
    print("Win percent: %f" %(games_won/(games_lost+games_won)))
    # return game_historiesp1, game_historiesp2
