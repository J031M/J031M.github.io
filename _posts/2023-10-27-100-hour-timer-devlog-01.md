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

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Do stuff

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
sys.exit()
```

First, let's start with the most important element, the organism. I've coded it as a seperate python module that I'll import into the main file.

Here's the code for `organism.py`

```python

from keras import Sequential
import numpy as np
import random
from keras import layers
from math import cos,sin,radians
import pygame

class organism:
    normprob = lambda self,x: x*2-1
    divdist = lambda self,x,settings: x/(settings['x_max']**2+settings['y_max']**2)**.5

    def __init__(self,settings,name=None):
        self.brain = Sequential([
            layers.Input(shape=(1,1)),
            layers.Dense(5,activation='tanh',name='l1'),
            layers.Dense(5,activation='tanh',name='l2'),
            layers.Dense(2,activation='tanh',name='l3')
            ])

        self.x = random.uniform(settings['x_min'],settings['x_max'])
        self.y = random.uniform(settings['y_min'],settings['y_max'])

        self.r = random.uniform(0,360)
        self.v = random.uniform(0,settings['v_max']//2)  
        self.dv = random.uniform(-settings['dv_max'], settings['dv_max'])

        self.health = 700
        self.d_food = 100   
        self.r_food = 0    
        self.fitness = 0  

        self.name = name

    def move(self,settings):
        input_data = np.array([[[self.normprob(self.divdist(self.d_food,settings))]]])
        nndr,nndv = [x.numpy() for x in self.brain(input_data)[0][0]]

        self.r += nndr * settings['dr_max'] * settings['dt']
        self.r %= 360

        self.v += nndv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']

        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy

    def layEgg(self,mutation_rate):
        egg = Egg(self,mutation_rate)
        return egg

    def draw(self, screen, width, height):
        rotated_rect = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(rotated_rect, (255,255,255), (0, 0, width, height))
        rotated_rect = pygame.transform.rotate(rotated_rect, -self.r)
        new_rect = rotated_rect.get_rect(center=(self.x, self.y))
        screen.blit(rotated_rect, new_rect.topleft)

class Egg:
    def __init__(self,parent,mutation_rate):
        self.laid = pygame.time.get_ticks()
        self.brain = parent.brain
        self.x = parent.x
        self.y = parent.y
        self.mutation_rate = mutation_rate

    def hatch(self,settings):
        egg_organism = organism(settings)
        egg_organism.brain = self.brain
        egg_organism.x = self.x
        egg_organism.y = self.y

        for layer in egg_organism.brain.layers:
            weights = layer.get_weights()
            x,y = weights[0].shape
            for m in range(x):
                for n in range(y):
                    if random.random() <= self.mutation_rate:
                        weights[0][m][n] += random.gauss(0,.04)
            layer.set_weights(weights)

        # ADD capability for neural network architecture to change

        return egg_organism

    def draw(self,screen):
        pygame.draw.circle(screen, (100, 100, 0), (self.x, self.y), 7)
```

got add comments
