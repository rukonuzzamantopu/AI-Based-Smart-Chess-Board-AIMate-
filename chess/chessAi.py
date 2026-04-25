'''
    Chess AI using Minimax with Alpha-Beta Pruning
    Keep DEPTH <= 4 for reasonable performance.
    DEPTH = 4 means the AI looks 4 half-moves ahead.
'''

import random

pieceScore = {'K': 0, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'p': 1}

knightScores = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]]

bishopScores = [
    [4, 3, 2, 1, 1, 2, 3, 4],
    [3, 4, 3, 2, 2, 3, 4, 3],
    [2, 3, 4, 3, 3, 4, 3, 2],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [2, 3, 4, 3, 3, 4, 3, 2],
    [3, 4, 3, 2, 2, 3, 4, 3],
    [4, 3, 2, 1, 1, 2, 3, 4]]

queenScores = [
    [1, 1, 1, 3, 1, 1, 1, 1],
    [1, 2, 3, 3, 3, 1, 1, 1],
    [1, 4, 3, 3, 3, 4, 2, 1],
    [1, 2, 3, 3, 3, 2, 2, 1],
    [1, 2, 3, 3, 3, 2, 2, 1],
    [1, 4, 3, 3, 3, 4, 2, 1],
    [1, 1, 2, 3, 3, 1, 1, 1],
    [1, 1, 1, 3, 1, 1, 1, 1]]

rookScores = [
    [4, 3, 4, 4, 4, 4, 3, 4],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 1, 2, 2, 2, 2, 1, 1],
    [4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 2, 1, 1, 2, 3, 4]]

whitePawnScores = [
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [5, 6, 6, 7, 7, 6, 6, 5],
    [2, 3, 3, 5, 5, 3, 3, 2],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]]

blackPawnScores = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 2, 3, 3, 2, 1, 1],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [2, 3, 3, 5, 5, 3, 3, 2],
    [5, 6, 6, 7, 7, 6, 6, 5],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8, 8]]

piecePositionScores = {
    'N': knightScores,
    'B': bishopScores,
    'Q': queenScores,
    'R': rookScores,
    'wp': whitePawnScores,
    'bp': blackPawnScores
}

CHECKMATE = 1000
STALEMATE = 0
DEPTH = 3  # Lowered to 3 for better responsiveness; raise to 4 for stronger play


def findRandomMoves(validMoves):
    return validMoves[random.randint(0, len(validMoves) - 1)]


def findBestMove(gs, validMoves, returnQueue):
    '''
    Entry point for the AI. Finds the best move using Minimax + Alpha-Beta.
    Puts the result into returnQueue for multiprocessing.
    '''
    global nextMove, whitePawnScores, blackPawnScores
    nextMove = None
    random.shuffle(validMoves)

    # For flipped board, swap pawn score tables
    if gs.playerWantsToPlayAsBlack:
        whitePawnScores, blackPawnScores = blackPawnScores, whitePawnScores

    if gs.whiteToMove:
        findMaxMove(gs, validMoves, DEPTH, -CHECKMATE, CHECKMATE)
    else:
        findMinMove(gs, validMoves, DEPTH, -CHECKMATE, CHECKMATE)

    # Restore pawn scores in case they were swapped
    if gs.playerWantsToPlayAsBlack:
        whitePawnScores, blackPawnScores = blackPawnScores, whitePawnScores

    returnQueue.put(nextMove)


def findMaxMove(gs, validMoves, depth, alpha, beta):
    global nextMove
    if depth == 0:
        return scoreBoard(gs)

    maxScore = -CHECKMATE
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        score = findMinMove(gs, nextMoves, depth - 1, alpha, beta)
        gs.undoMove()
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
                print(move, score)
        alpha = max(alpha, maxScore)
        if alpha >= beta:
            break
    return maxScore


def findMinMove(gs, validMoves, depth, alpha, beta):
    global nextMove
    if depth == 0:
        return scoreBoard(gs)

    minScore = CHECKMATE
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        score = findMaxMove(gs, nextMoves, depth - 1, alpha, beta)
        gs.undoMove()
        if score < minScore:
            minScore = score
            if depth == DEPTH:
                nextMove = move
                print(move, score)
        beta = min(beta, minScore)
        if beta <= alpha:
            break
    return minScore


def scoreBoard(gs):
    '''
    Evaluate the board from White's perspective.
    '''
    if gs.checkmate:
        if gs.whiteToMove:
            return -CHECKMATE
        else:
            return CHECKMATE
    if gs.stalemate:
        return STALEMATE

    score = 0
    for row in range(8):
        for col in range(8):
            square = gs.board[row][col]
            if square == '--':
                continue
            piecePositionScore = 0
            if square[1] != 'K':
                if square[1] == 'p':
                    piecePositionScore = piecePositionScores[square][row][col]
                else:
                    piecePositionScore = piecePositionScores[square[1]][row][col]
            if square[0] == 'w':
                score += pieceScore[square[1]] + piecePositionScore * 0.1
            elif square[0] == 'b':
                score -= pieceScore[square[1]] + piecePositionScore * 0.1
    return score