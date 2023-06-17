import pygame
from random import randint
from collections import namedtuple
from enum import Enum
import numpy as np
import copy

pygame.init()

font = pygame.font.SysFont('arial', 20)

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 2000
DELAY_FOR_TIMEOUTMOVE = 200

class Player:
    def __init__(self, id, color, x, y, size_w, size_h):
        self.actPos = Point(x, y)
        self.id = id
        self.lastMove = Action(id, randint(1,4))
        self.color = color
        # Store Window Size
        self.len_w = size_w
        self.len_h = size_h

    def _getNewPos(self, action):
        STEP = 1
        # Move Left
        if action == 1:
            self.actPos = Point((self.actPos.x + STEP) % self.len_w, self.actPos.y)
        # Move Right
        elif action == 2:
            self.actPos = Point((self.actPos.x - STEP) % self.len_w, self.actPos.y)
        # Move Down
        elif action == 3:
            self.actPos = Point(self.actPos.x, (self.actPos.y + STEP) % self.len_h)
        # Move Up
        elif action == 4:
            self.actPos = Point(self.actPos.x, (self.actPos.y - STEP) % self.len_h)
        else:
            return False
        return self.actPos

    def __repr__(self):
        return f"PlayerOBJ - ID{self.id}, ActPos:{self.actPos}"

class Action:
    def __init__(self, id, action):
        self.playerID = id
        self.action = action

    def __repr__(self):
        return (f'ActionOBJ - ID:{self.playerID}, Action:{self.action}')

class TronGame:

    def __init__(self, blockAmount_w=20, blockAmount_h=20, useTimeout=True, learingType=1):
        self.screen_w = blockAmount_w * BLOCK_SIZE
        self.screen_h = blockAmount_h * BLOCK_SIZE

        # init display
        self.display = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption('Tron')
        self.clock = pygame.time.Clock()

        # init vars
        self.lastMoveUpdate = 0
        self.amountMoves = 0
        self.gamefield_w = blockAmount_w
        self.gamefield_h = blockAmount_h
        self.gamefield = [[0 for x in range(self.gamefield_w)] for y in range(self.gamefield_h)]
        self.act_players = []
        self.queuedActions = []
        self.learningType = learingType
        self.players = []
        self.useTimeout = useTimeout
        self.reset()

    def reset(self):
        self.act_players = []
        self.frame_iteration = 0
        self.amountMoves = 0
        self.gamefield = [[0 for x in range(self.gamefield_w)] for y in range(self.gamefield_h)]
        self.queuedActions = []
        self.act_players = self.players.copy()

    # Register a Action for a Player
    def registerAction(self, id, action):
        if action > 4 or action <1:
            return False
        validID = False
        for p in self.act_players:
            if p.id == id:
                validID = True
        if validID:
            validActionRegister = True
            for a in self.queuedActions:
                if a.id == id:
                    validActionRegister = False
            if validActionRegister:
                self.queuedActions.append(Action(id, action))
                return True
            else:
                return False
        else:
            return False

    # Register new Player for
    def registerPlayer(self, color=(randint(0,255), randint(0,255), randint(0,255))):
        # Generate a new ID
        id = randint(0, 1000)
        while id in self.act_players:
            id = randint(0, 1000)
        player = Player(id, color, randint(0, self.gamefield_w), randint(0, self.gamefield_h), self.gamefield_w, self.gamefield_h)
        self.act_players.append(player)
        self.players.append(player)
        return id

    # Move one Player
    def _movePlayer(self, p, act):
        newPos = p._getNewPos(int(act.action))
        if p.lastMove != act:
            p.lastMove = act
        kill = self._checkForKill(p, newPos)
        if kill:
            self.act_players.remove(p)
        else:
            self.gamefield[newPos.y-1][newPos.x-1] = p.id

    # Move all Player when all action are received
    def _move_Players(self):
        if len(self.queuedActions) == len(self.act_players):
            self.lastMoveUpdate = pygame.time.get_ticks()
            for act in self.queuedActions:
                for p in self.act_players:
                    if p.id == act.playerID:
                        self._movePlayer(p, act)
                        break

            self.queuedActions.clear()
            self.amountMoves += 1
            return True
        else:
            # Check for TimeOut-Move
            if self.useTimeout:
                if pygame.time.get_ticks() - self.lastMoveUpdate >= DELAY_FOR_TIMEOUTMOVE:
                    for p in self.act_players:
                        moved = False
                        # Try to find Registered Action
                        for act in self.queuedActions:
                            if p.id == act.playerID:
                                self._movePlayer(p, act)
                                moved = True
                                break
                        print("USE TIMEOUT")
                        self.queuedActions.clear()

                        if not moved:
                            # Player TimedOut -> use old Move
                            self._movePlayer(p, p.lastMove)

                    self.amountMoves += 1
                    return True
            return False

    # Return the Playerobj. for a given ID
    def find_player_by_id(self, player_id):
        for player in self.act_players:
            if player.id == player_id:
                return player
        return None

    # Check if the player is now defeated
    def _checkForKill(self, p, newPos):
        if self.gamefield[newPos.y-1][newPos.x-1] != 0:
            # Remove Player from gamefield
            for x in range(len(self.gamefield)):
                for y in range(len(self.gamefield[x])):
                    if self.gamefield[x][y] == p.id:
                        self.gamefield[x][y] = 0
            return True
        return False

    # Render the Output
    def _render(self):
        self.display.fill(BLACK)
        # Draw Gamefield
        for x in range(self.gamefield_w):
            for y in range(self.gamefield_h):
                if self.gamefield[y][x] != 0:
                    ply = self.find_player_by_id(self.gamefield[y][x])
                    if ply != None:
                        pygame.draw.rect(self.display, ply.color, pygame.Rect(x * BLOCK_SIZE + 4, y * BLOCK_SIZE + 4, 12, 12))
        # Draw Score:
        text = font.render("Player: " + str(len(self.act_players)) + " Moves:" + str(self.amountMoves), True, WHITE)
        self.display.blit(text, [0,0])

        # Update Screen
        pygame.display.flip()

    # Check for GameOver
    def _checkGameOver(self):
        # TODO: Better Win Condition (1 player left when not in single player mode)
        if len(self.act_players) == 0:
            return True
        return False

    # Generate State for ML
    def getState(self, type=1,  playerID=-1,):
        # Player Pos and Gamefield
        if type == 1:
            norm_gamefield = np.array(self.gamefield, dtype=int)
            for x in range(len(self.gamefield)):
                for y in range(len(self.gamefield[x])):
                    if self.gamefield[x][y] != 0:
                        norm_gamefield[x][y] = 1

            if playerID == -1:
                try:
                    playerID = self.act_players[0].id
                    player = self.find_player_by_id(playerID)
                    playerPos = [player.actPos.x, player.actPos.y]
                except:
                    playerPos = [-1,-1]
            returnState = np.concatenate((playerPos, norm_gamefield.flatten()), axis=0)
            return returnState.flatten()

        # Only Surrounding Field
        if type == 2:
            player = self.find_player_by_id(playerID)
            if player == None and len(self.act_players):
                player = self.act_players[0]
            if player:
                actPos = player.actPos
                # Field in a 3x3 area with actpos in center
                y = actPos.y - 1
                x = actPos.x - 1
                surrounding = [
                    [self.gamefield[(y - 1) % self.gamefield_h][(x - 1) % self.gamefield_w], self.gamefield[(y - 1) % self.gamefield_h][x % self.gamefield_w], self.gamefield[(y - 1) % self.gamefield_h][(x + 1) % self.gamefield_w]],
                    [self.gamefield[y % self.gamefield_h][(x - 1) % self.gamefield_w], self.gamefield[y % self.gamefield_h][x % self.gamefield_w], self.gamefield[y % self.gamefield_h][(x + 1) % self.gamefield_w]],
                    [self.gamefield[(y + 1) % self.gamefield_h][(x - 1) % self.gamefield_w], self.gamefield[(y + 1) % self.gamefield_h][x % self.gamefield_w], self.gamefield[(y + 1) % self.gamefield_h][(x + 1) % self.gamefield_w]]
                ]
                norm_surrounding = np.array(surrounding, dtype=int)
                for x in range(len(surrounding)):
                    for y in range(len(surrounding[x])):
                        if surrounding[x][y] != 0:
                            norm_surrounding[x][y] = 1
                returnState = norm_surrounding.flatten()
                return returnState
            else:
                return [1,1,1,1,1,1,1,1,1]

    # Do one Game Step
    def game_step(self):
        reward = 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Do all Movements
        self._move_Players()

        # Do new Frame rendering
        self.frame_iteration += 1
        self._render()
        self.clock.tick(SPEED)

        # Check if game Over
        if self._checkGameOver():
            reward = -1
            return self.getState(type=self.learningType), reward, True

        return self.getState(type=self.learningType), reward, False

    # Step function for learning the ML-Net
    def step(self, action):
        self.registerAction(self.act_players[0].id, action)
        return self.game_step()
