Snake NEAT Neuroevolution Project
=================================

This project implements a NEAT-style neuroevolution system to train neural networks to play the Snake game autonomously. It uses pygame for visualization and simulates many games in parallel for faster evolution.

Overview
--------
- **NEATNetwork, Node, Connection**: Classes to represent a neural network with evolving topology (nodes and connections), supporting mutations and serialization.
- **SnakeGame**: Simulates the snake game, providing inputs to the neural network and updating the game state based on the network's outputs.
- **Population**: Manages a group of neural networks, runs simulations, tracks fitness, performs selection, mutation, and speciation, and handles saving/loading the best genome.
- **Main Loop**: Runs the simulation, handles user input for evolving, saving/loading, and adjusting simulation speed.

NEAT-style Neural Network
-------------------------
- Each network consists of input, hidden, and output nodes connected by weighted edges.
- Networks mutate by adding nodes/connections and changing weights.
- The network receives game state inputs (distances to walls, collision detection, food direction) and outputs movement decisions (up, down, left, right).

Snake Game Simulation
---------------------
- Multiple games run in parallel, each controlled by a different neural network.
- The snake receives sensory inputs and the neural network decides its movement.
- Fitness is calculated based on score, survival time, and proximity to food.

Population Management
---------------------
- Networks are evolved using selection, mutation, and speciation.
- The best-performing network is saved and can be loaded for future runs.
- Visualization includes drawing all games and the best network's structure.

Controls
--------
- SPACE: Force evolution to next generation
- S: Save the best genome to file
- L: Load the best genome from file
- '+' / =: Increase simulation speed
- -: Decrease simulation speed

Demo
----
A demonstration video showing the NEAT-evolved neural network playing Snake is included below:

[![Demo Video](Demo.mp4)](Demo.mp4)

Files
-----
- `snake_nn.py`: Main code for the NEAT Snake neuroevolution system.
- `Demo.mp4`: Demo video of the trained agent.
- `best_snake_genome.json`: Saved best genome (created during training).

Requirements
------------
- Python 3.x
- pygame
- numpy

To run:
-------
1. Install requirements: `pip install pygame numpy`
2. Run the main script: `python snake_nn.py`
3. Watch the evolution and interact using the controls above.

How it works
------------
- Each neural network controls a snake in its own simulation.
- The network receives 10 inputs: distances to walls, collision detection in 4 directions, and food direction.
- The network outputs 4 values, corresponding to up, down, left, right movement.
- Networks evolve over generations, improving their ability to survive and eat food.

Enjoy watching neuroevolution in action!
