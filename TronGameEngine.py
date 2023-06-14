import pygame
from random import randint
from collections import namedtuple
from enum import Enum

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
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10
DELAY_FOR_TIMEOUTMOVE = 5
class Player:
    def __init__(self, id, x, y, win_w, win_h):
        self.actPos = Point(x,y)
        self.id = id
        self.trace = []
        self.lastMove = Action(id, randint(1,4))
        # Store Window Size
        self.win_w = win_w
        self.win_h = win_h

    def _move(self, action):
        self.trace.append(self.actPos)
        # Move Left
        if action == 1:
            self.actPos = Point((self.actPos.x + BLOCK_SIZE) % self.win_w, self.actPos.y)
        # Move Right
        elif action == 2:
            self.actPos = Point((self.actPos.x - BLOCK_SIZE) % self.win_w, self.actPos.y)
        # Move Down
        elif action == 3:
            self.actPos = Point(self.actPos.x, (self.actPos.y + BLOCK_SIZE) % self.win_h)
        # Move Up
        elif action == 4:
            self.actPos = Point(self.actPos.x, (self.actPos.y - BLOCK_SIZE) % self.win_h)
        else:
            return False

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

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = w
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tron')
        self.clock = pygame.time.Clock()
        self.lastAction = 0
        self.players = []
        self.actions = []
        self.reset()

    def reset(self):
        # TODO Reset Players
        self.frame_iteration = 0
        self.actions = []

    # Register a Action for a Player
    def registerAction(self, id, action):
        if action > 4 or action <1:
            return False
        validIDd = False
        for p in self.players:
            if p.id == id:
                validID = True
        if validID:
            validActionRegister = True
            for a in self.actions:
                if a.id == id:
                    validActionRegister = False
            if validActionRegister:
                self.actions.append(Action(id, action))
                return True
            else:
                return False
        else:
            return False

    # Register new Player for
    def registerPlayer(self):
        id = randint(0, 1000)
        while id in self.players:
            id = randint(0, 1000)
        # TODO dont use fixval as start
        self.players.append(Player(id, 50,50, self.w, self.h))
        return id

    # Move one Player
    def _movePlayer(self, p, act):
        newPos = p._move(int(act.action))
        if p.lastMove != act:
            p.lastMove = act
        kill = self._checkForKill(p, newPos)
        if kill:
            self.players.remove(p)

    # Move all Player when all action are recived
    def _move_Players(self):
        if len(self.actions) == len(self.players):
            self.lastAction = pygame.time.get_ticks()
            for act in self.actions:
                for p in self.players:
                    if p.id == act.playerID:
                        self._movePlayer(p, act)
                        break

            self.actions.clear()
            return True
        else:
            # Check for TimeOut-Move
            if pygame.time.get_ticks() - self.lastAction >= DELAY_FOR_TIMEOUTMOVE:
                for p in self.players:
                    moved = False
                    # Try to find Registerd Action
                    for act in self.actions:
                        if p.id == act.playerID:
                            self._movePlayer(p, act)
                            moved = True
                            break

                    if not moved:
                        # Player TimedOut -> use old Move
                        self._movePlayer(p, p.lastMove)
                return True
            return False




    # Check if the player is now defeated
    def _checkForKill(self, p, newPos):
        for ply in self.players:
            if ply != p:
                if ply.actPos == newPos:
                    return True
            for t in ply.trace:
                if t == newPos:
                    return True
        return False

    # Render the Output
    def _render(self):
        self.display.fill(BLACK)

        # Draw all player
        for p in self.players:
            # Draw Head of a Player
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(p.actPos.x, p.actPos.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(p.actPos.x + 4, p.actPos.y + 4, 12,12))
            # Draw Trace of a Player
            for trace in p.trace:
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(trace.x + 4, trace.y + 4, 12, 12))

        # Draw Score:
        text = font.render("Score: " + str(self.frame_iteration), True, WHITE)
        self.display.blit(text, [0,0])

        # Update Screen
        pygame.display.flip()

    # Check for GameOver
    def _checkGameOver(self):
        # TODO: Better Win Condition (1 player left when not in single player mode)
        if(len(self.players) == 0):
            return True
        return False

    # Do one Game Step
    def game_step(self):
        print("gameStep")
        # End game based on event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        print("events")
        # Do all Movements
        self._move_Players()
        print("move")
        # Check if game Over
        if(self._checkGameOver()):
            return False
        print("GameOver")
        # Do new Frame rendering
        self.frame_iteration += 1
        self._render()
        print("render")
        self.clock.tick(SPEED)

        return True