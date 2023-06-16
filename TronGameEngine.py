import pygame
from random import randint
from collections import namedtuple
from enum import Enum
import numpy as np
import copy

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10
DELAY_FOR_TIMEOUTMOVE = 5

class Player:
    def __init__(self, id, color, x, y, size_w, size_h):
        self.actPos = Point(x,y)
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
        print(self.actPos)
        return(self.actPos)

    def __repr__(self):
        return f"PlayerOBJ - ID{self.id}, ActPos:{self.actPos}"

class Action:
    def __init__(self, id, action):
        self.playerID = id
        self.action = action

    def __repr__(self):
        return (f'ActionOBJ - ID:{self.playerID}, Action:{self.action}')

class TronGame:

    def __init__(self, blockAmount_w=20, blockAmount_h=20):
        self.screen_w = blockAmount_w * BLOCK_SIZE
        self.screen_h = blockAmount_h * BLOCK_SIZE

        # init display
        self.display = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption('Tron')
        self.clock = pygame.time.Clock()

        # init vars
        self.lastMoveUpdate = 0
        self.gamefield_w = blockAmount_w
        self.gamefield_h = blockAmount_h
        self.gamefield = [[0 for x in range(self.gamefield_w)] for y in range(self.gamefield_h)]
        self.players = []
        self.queuedActions = []
        self.reset()

    def reset(self):
        # TODO Reset Players
        print("Engine: reset")
        self.players = []
        self.frame_iteration = 0
        self.gamefield = [[0 for x in range(self.gamefield_w)] for y in range(self.gamefield_h)]
        self.queuedActions = []


    # Register a Action for a Player
    def registerAction(self, id, action):
        if action > 4 or action <1:
            return False
        validID = False
        for p in self.players:
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
    def registerPlayer(self, color = (randint(0,255), randint(0,255), randint(0,255))):
        # Generate a new ID
        id = randint(0, 1000)
        while id in self.players:
            id = randint(0, 1000)

        self.players.append(Player(id, color, randint(0, self.gamefield_w), randint(0, self.gamefield_h), self.gamefield_w, self.gamefield_h))
        return id

    # Move one Player
    def _movePlayer(self, p, act):
        newPos = p._getNewPos(int(act.action))
        if p.lastMove != act:
            p.lastMove = act
        kill = self._checkForKill(p, newPos)
        if kill:
            self.players.remove(p)
        else:
            self.gamefield[newPos.x][newPos.y] = p.id

    # Move all Player when all action are recived
    def _move_Players(self):
        if len(self.queuedActions) == len(self.players):
            self.lastMoveUpdate = pygame.time.get_ticks()
            for act in self.queuedActions:
                for p in self.players:
                    if p.id == act.playerID:
                        self._movePlayer(p, act)
                        break

            self.queuedActions.clear()
            return True
        else:
            # Check for TimeOut-Move
            if pygame.time.get_ticks() - self.lastMoveUpdate >= DELAY_FOR_TIMEOUTMOVE:
                for p in self.players:
                    moved = False
                    # Try to find Registerd Action
                    for act in self.queuedActions:
                        if p.id == act.playerID:
                            self._movePlayer(p, act)
                            moved = True
                            break
                    self.queuedActions.clear()
                    if not moved:
                        # Player TimedOut -> use old Move
                        self._movePlayer(p, p.lastMove)
                return True
            return False

    # Return the Playerobj. for a given ID
    def find_player_by_id(self, player_id):
        for player in self.players:
            if player.id == player_id:
                return player
        return None

    # Check if the player is now defeated
    def _checkForKill(self, p, newPos):
        if self.gamefield[newPos.x][newPos.y] != 0:
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
        # Draw all player
        for x in range(len(self.gamefield)):
            for y in range(len(self.gamefield[x])):
                if self.gamefield[x][y] != 0:
                    ply = self.find_player_by_id(self.gamefield[x][y])
                    if ply != None:
                        pygame.draw.rect(self.display, ply.color, pygame.Rect(x * BLOCK_SIZE + 4, y * BLOCK_SIZE + 4, 12, 12))
        # Draw Score:
        text = font.render("Player: " + str(len(self.players)) + " Frame:" + str(self.frame_iteration), True, WHITE)
        self.display.blit(text, [0,0])

        # Update Screen
        pygame.display.flip()

    # Check for GameOver
    def _checkGameOver(self):
        # TODO: Better Win Condition (1 player left when not in single player mode)
        if(len(self.players) == 0):
            return True
        return False

    # Generate State for ML
    def getState(self):
        norm_gamefield = np.array(self.gamefield,dtype=int)
        for x in range(len(self.gamefield)):
            for y in range(len(self.gamefield[x])):
                if self.gamefield[x][y] != 0:
                    norm_gamefield[x][y] = 1
        return norm_gamefield

    # Do one Game Step
    def game_step(self):
        reward = 1
        # End game based on event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Do all Movements
        self._move_Players()

        # Check if game Over
        if(self._checkGameOver()):
            reward = -10
            return reward, False, self.frame_iteration

        # Do new Frame rendering
        self.frame_iteration += 1
        self._render()
        self.clock.tick(SPEED)

        return reward, True, self.frame_iteration