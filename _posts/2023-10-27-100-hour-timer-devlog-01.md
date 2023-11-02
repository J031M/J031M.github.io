---
title: Setting up an evolution simulation using Pygame [devlog 1]
date: 2023-10-27 11:16:00
tags: [devlog,100timer]     # TAG names should always be lowercase
toc: true
---

This project is available [here](https://github.com/J031M/Evolution_Simulation_PyGame) if you'd like to check it out.

There aren't many animations that will remain unique over 100 hours. And since I love AI, I chose to have my animation be an evolution simulation. I will port it to unity eventually but just to get the prototype out, I'm using Pygame. I'd say this has gotten off to a pretty great start!

![](/assets/img/misc/evosim.gif)
_the squares are the organisms, white circles are food and green circles are eggs_

Pygame runs one loop continuously and all the game updates happens in that one loop. Here's the basic structure of the pygame project.

```python
import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolution Simulation")

# Colors
WHITE = (255, 255, 255)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw your organisms, food, and other game elements here

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
sys.exit()
```


