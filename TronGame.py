import TronGameEngine
from pynput import keyboard
import numpy as np

# setup
game = TronGameEngine.TronGame(30, 30)

def on_key_press(key):
    print("key pressed")
    try:
        if key.char == 'w':
            game.registerAction(p1, 3)
        elif key.char == 'a':
            game.registerAction(p1,1)
        elif key.char == 's':
            game.registerAction(p1, 2)
        elif key.char == 'd':
            game.registerAction(p1, 0)
    except AttributeError:
        pass

p1 = game.registerPlayer((0,0,255))
game.registerPlayer()
print(f'Player1 ID: {p1}')
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

#p2 = game.registerPlayer()
#print(f'Player2 ID: {p2}')

running = True
counter = 0
# game loop
while running:
    # Get Player Movement
    # use Eventlistener for p1
    # Do GameStep
    counter +=1
    if counter%10==0:
        game.registerPlayer()
    running = game.game_step()
    game.getState()

exit(0)
listener.join()