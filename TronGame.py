import TronGameEngine
from pynput import keyboard

# setup
game = TronGameEngine.TronGame()

def on_key_press(key):
    try:
        if key.char == 'w':
            game.registerAction(p1, 4)
        elif key.char == 'a':
            game.registerAction(p1,2)
        elif key.char == 's':
            game.registerAction(p1, 3)
        elif key.char == 'd':
            game.registerAction(p1, 1)
    except AttributeError:
        pass




p1 = game.registerPlayer()
print(f'Player1 ID: {p1}')
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

#p2 = game.registerPlayer()
#print(f'Player2 ID: {p2}')

# game loop
while True:
    # Get Player Movement
    # use Eventlistener for p1
    # Do GameStep
    game.game_step()

listener.join()