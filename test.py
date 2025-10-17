import pyautogui
import time

print("Move mouse in 3 seconds...")
time.sleep(3)

pyautogui.moveTo(500, 500, duration=1)  # move to x=500, y=500
print("Done")