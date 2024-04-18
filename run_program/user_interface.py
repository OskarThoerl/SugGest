"""
UI Class by Oskar Thörl and Nadja Bobić

This is the UI (User Interface) class. It handles all elements and interactions around the user interface of the program.
It uses the Pygame library to create a full-screen display, load and display images, play sounds.

The class contains several methods:

The __init__ method initializes Pygame, gets the screen size, creates a full-screen canvas, loads and scales images,
    initializes the Pygame mixer module, and loads sounds.
The display method displays an image and plays a corresponding sound based on a detected gesture.
    It calculates the position to center the image, draws the image on the canvas, updates the display, plays the sound,
    waits for 5 seconds, and then clears the canvas.
The clear method clears the screen by filling the canvas with black and updating the display.
The __del__ method defines what happens upon deletion of an instance of this class, which is to quit Pygame.
The functions of this UI class are called in the main class.
"""

# Initialize Libraries
import pygame as pg
from time import sleep

# Class that handles all the UI of the program
class UI:

    # Function to initialize all variables and load all media
    def __init__(self):
        # Initialize Pygame
        pg.init()

        # Get the screen size
        self.screen_info = pg.display.Info()
        self.screen_width = self.screen_info.current_w
        self.screen_height = self.screen_info.current_h

        # Create a Pygame surface (canvas)
        self.canvas = pg.display.set_mode((0, 0), pg.FULLSCREEN)

        # Load images
        consume = pg.image.load("media/images/consume.png")
        sleep = pg.image.load("media/images/sleep.png")
        attention = pg.image.load("media/images/attention.png")

        # Get max height / width depending on h/w ratio
        if self.screen_width > self.screen_height:
            size = self.screen_height
        else:
            size = self.screen_width

        # Scale images to a consistent size
        consume = pg.transform.scale(consume, (size, size))
        sleep = pg.transform.scale(sleep, (size, size))
        attention = pg.transform.scale(attention, (size, size))

        # Add images to image dictionary
        self.images = {
            "consume": consume,
            "sleep": sleep,
            "attention": attention
        }

        # Initialize the mixer module
        pg.mixer.init()

        # Load sounds
        consume_sound = pg.mixer.Sound("media/audio/consume_f.mp3")
        sleep_sound = pg.mixer.Sound("media/audio/sleep_f.mp3")
        attention_sound = pg.mixer.Sound("media/audio/attention_f.mp3")

        # Add sounds to sound dictionary
        self.sounds = {
            "consume": consume_sound,
            "sleep": sleep_sound,
            "attention": attention_sound
        }

    # Function to display image and play sound according to a gesture that was detected
    def display(self, image):
        # Calculate the position to center the image
        x = (self.screen_width - self.images[image].get_width()) // 2
        y = (self.screen_height - self.images[image].get_height()) // 2

        # Draw the image on the canvas
        self.canvas.blit(self.images[image], (x, y))

        # Update the display
        pg.display.flip()

        # Play the fitting sound
        self.sounds[image].play()

        # Wait for 5 seconds
        sleep(5)

        # Clear the canvas
        self.clear()

    # Function to clear the screen
    def clear(self):
        # Fill the canvas with black
        self.canvas.fill((0, 0, 0))

        # Update the display
        pg.display.flip()

    # Function to define what happens upon deletion of an instance of this class
    def __del__(self):
        # Quit Pygame
        pg.quit()
