import pygame
import numpy as np
import random
import time
import copy
import json
import os
from collections import defaultdict

# --- NEAT-style Neural Network ---
class Connection:
    def __init__(self, input_node, output_node, weight=None, enabled=True):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight if weight is not None else random.uniform(-1, 1)
        self.enabled = enabled
        self.innovation = None

    def to_dict(self):
        return {
            'input_node': self.input_node,
            'output_node': self.output_node,
            'weight': self.weight,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['input_node'], data['output_node'], data['weight'], data['enabled'])

class Node:
    def __init__(self, node_id, node_type, x_pos=0, y_pos=0):
        self.node_id = node_id
        self.node_type = node_type
        self.value = 0
        self.x_pos = x_pos
        self.y_pos = y_pos

    def to_dict(self):
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'x_pos': self.x_pos,
            'y_pos': self.y_pos
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['node_id'], data['node_type'], data['x_pos'], data['y_pos'])

class NEATNetwork:
    def __init__(self, input_size=10, output_size=4):  # Increased input size for boundary awareness
        self.nodes = {}
        self.connections = []
        self.input_size = input_size
        self.output_size = output_size
        self.next_node_id = 0
        self.fitness = 0
        
        # Create input nodes
        for i in range(input_size):
            node = Node(self.next_node_id, 'input', x_pos=0, y_pos=i)
            self.nodes[self.next_node_id] = node
            self.next_node_id += 1
        
        # Create output nodes
        for i in range(output_size):
            node = Node(self.next_node_id, 'output', x_pos=2, y_pos=i)
            self.nodes[self.next_node_id] = node
            self.next_node_id += 1

    def to_dict(self):
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'connections': [c.to_dict() for c in self.connections],
            'input_size': self.input_size,
            'output_size': self.output_size,
            'next_node_id': self.next_node_id,
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, data):
        net = cls(data['input_size'], data['output_size'])
        net.nodes = {int(k): Node.from_dict(v) for k, v in data['nodes'].items()}
        net.connections = [Connection.from_dict(c) for c in data['connections']]
        net.next_node_id = data['next_node_id']
        net.fitness = data['fitness']
        return net

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self):
        """Create a deep copy of this network"""
        new_net = NEATNetwork(self.input_size, self.output_size)
        new_net.nodes = {}
        new_net.connections = []
        new_net.next_node_id = self.next_node_id
        
        # Copy nodes
        for node_id, node in self.nodes.items():
            new_node = Node(node.node_id, node.node_type, node.x_pos, node.y_pos)
            new_net.nodes[node_id] = new_node
        
        # Copy connections
        for conn in self.connections:
            new_conn = Connection(conn.input_node, conn.output_node, conn.weight, conn.enabled)
            new_net.connections.append(new_conn)
        
        return new_net

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs):
        # Reset all node values
        for node in self.nodes.values():
            node.value = 0
        
        # Set input values
        input_nodes = [n for n in self.nodes.values() if n.node_type == 'input']
        for i, node in enumerate(input_nodes):
            if i < len(inputs):
                node.value = inputs[i]
        
        # Evaluate network
        max_iterations = 10
        for _ in range(max_iterations):
            for conn in self.connections:
                if conn.enabled and conn.input_node in self.nodes and conn.output_node in self.nodes:
                    input_node = self.nodes[conn.input_node]
                    output_node = self.nodes[conn.output_node]
                    if output_node.node_type != 'input':
                        output_node.value += input_node.value * conn.weight
        
        # Apply activation to non-input nodes
        for node in self.nodes.values():
            if node.node_type != 'input':
                node.value = self.sigmoid(node.value)
        
        # Get output values
        output_nodes = [n for n in self.nodes.values() if n.node_type == 'output']
        outputs = [node.value for node in output_nodes]
        return outputs

    def add_connection_mutation(self):
        """Add a new connection between two nodes"""
        available_inputs = [n.node_id for n in self.nodes.values() 
                          if n.node_type in ['input', 'hidden']]
        available_outputs = [n.node_id for n in self.nodes.values() 
                           if n.node_type in ['hidden', 'output']]
        
        if not available_inputs or not available_outputs:
            return
        
        existing = [(c.input_node, c.output_node) for c in self.connections]
        
        attempts = 20
        for _ in range(attempts):
            input_id = random.choice(available_inputs)
            output_id = random.choice(available_outputs)
            
            if input_id != output_id and (input_id, output_id) not in existing:
                new_conn = Connection(input_id, output_id)
                self.connections.append(new_conn)
                break

    def add_node_mutation(self):
        """Add a new node by splitting an existing connection"""
        if not self.connections:
            return
        
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return
        
        old_conn = random.choice(enabled_conns)
        old_conn.enabled = False
        
        new_node = Node(self.next_node_id, 'hidden', x_pos=1, 
                       y_pos=len([n for n in self.nodes.values() if n.node_type == 'hidden']))
        self.nodes[self.next_node_id] = new_node
        
        conn1 = Connection(old_conn.input_node, self.next_node_id, weight=1.0)
        conn2 = Connection(self.next_node_id, old_conn.output_node, weight=old_conn.weight)
        
        self.connections.extend([conn1, conn2])
        self.next_node_id += 1

    def mutate_weights(self, rate=0.9, strength=0.4):  # Increased mutation rate and strength
        """Mutate existing connection weights"""
        for conn in self.connections:
            if random.random() < rate:
                if random.random() < 0.85:  # Slightly more complete resets
                    conn.weight += random.uniform(-strength, strength)
                else:
                    conn.weight = random.uniform(-3, 3)  # Wider reset range
                conn.weight = max(-6, min(6, conn.weight))  # Allow stronger weights

    def mutate(self):
        """Perform various mutations - increased rates"""
        if random.random() < 0.95:  # Increased from 0.7
            self.mutate_weights()
        
        if random.random() < 0.4:   # Increased from 0.15
            self.add_connection_mutation()
        
        if random.random() < 0.08:  # Increased from 0.02
            self.add_node_mutation()
        
        if random.random() < 0.15:  # Increased from 0.05
            if self.connections:
                conn = random.choice(self.connections)
                conn.enabled = not conn.enabled

# --- Snake Game ---
# === Fast Training == #
CELL_SIZE = 2  # Smaller cells to fit more simulations
FPS = 60*4
SIMS_PER_ROW = 20  # Increased from 6
SIMS_PER_COL = 20   # Increased from 4

# # === Individual Training === #
# CELL_SIZE = 20  # Smaller cells to fit more simulations
# FPS = 60
# SIMS_PER_ROW = 2  # Increased from 6
# SIMS_PER_COL = 2   # Increased from 4

GRID_WIDTH, GRID_HEIGHT = 20, 20  # Smaller grid
TOTAL_SIMS = SIMS_PER_ROW * SIMS_PER_COL  # Now 60 simulations
GAME_WIDTH = GRID_WIDTH * CELL_SIZE
GAME_HEIGHT = GRID_HEIGHT * CELL_SIZE

pygame.init()
WINDOW_WIDTH = SIMS_PER_ROW * GAME_WIDTH + 400
WINDOW_HEIGHT = SIMS_PER_COL * GAME_HEIGHT + 100
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)

class SnakeGame:
    def __init__(self, nn, x_offset=0, y_offset=0):
        self.nn = nn
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.reset()

    def reset(self):
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.dir = (1, 0)
        self.spawn_food()
        self.start_time = time.time()
        self.score = 0
        self.steps = 0
        self.alive = True
        self.fitness = 0
        self.steps_since_food = 0

    def spawn_food(self):
        attempts = 100
        for _ in range(attempts):
            food_pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food_pos not in self.snake:
                self.food = food_pos
                return
        self.food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))

    def get_inputs(self):
        head = self.snake[0]
        inputs = []
        
        # 1-4: Distance to walls (normalized)
        dist_to_left = head[0] / GRID_WIDTH
        dist_to_right = (GRID_WIDTH - 1 - head[0]) / GRID_WIDTH
        dist_to_top = head[1] / GRID_HEIGHT
        dist_to_bottom = (GRID_HEIGHT - 1 - head[1]) / GRID_HEIGHT
        inputs.extend([dist_to_left, dist_to_right, dist_to_top, dist_to_bottom])
        
        # 5-8: Collision detection in 4 directions
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        for dx, dy in directions:
            next_pos = (head[0] + dx, head[1] + dy)
            if (next_pos in self.snake or 
                next_pos[0] < 0 or next_pos[0] >= GRID_WIDTH or
                next_pos[1] < 0 or next_pos[1] >= GRID_HEIGHT):
                inputs.append(1.0)  # Collision detected
            else:
                inputs.append(0.0)  # Safe
        
        # 9-10: Food direction (normalized)
        food_dx = (self.food[0] - head[0]) / GRID_WIDTH
        food_dy = (self.food[1] - head[1]) / GRID_HEIGHT
        inputs.extend([food_dx, food_dy])
        
        return inputs

    def step(self):
        if not self.alive:
            return
            
        self.steps += 1
        self.steps_since_food += 1
        inputs = self.get_inputs()
        outputs = self.nn.forward(inputs)
        
        # Choose direction - require higher confidence for action
        max_output = max(outputs)
        if max_output > 0.6:
            action = outputs.index(max_output)
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
            new_dir = directions[action]
            
            # Prevent 180-degree turns
            if (new_dir[0] * -1, new_dir[1] * -1) != self.dir:
                self.dir = new_dir

        head = self.snake[0]
        new_head = (head[0] + self.dir[0], head[1] + self.dir[1])

        # More patient collision check
        max_steps_without_food = max(200, len(self.snake) * 50)
        max_total_steps = 2000
        
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
            self.steps_since_food > max_steps_without_food or
            self.steps > max_total_steps):
            
            self.alive = False
            # Enhanced fitness calculation
            distance_to_food = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            proximity_bonus = max(0, (50 - distance_to_food) * 5)
            
            self.fitness = (self.score * 2000 + 
                          self.steps * 2 + 
                          proximity_bonus - 
                          self.steps_since_food * 3)
            
            self.nn.fitness = max(0, self.fitness)
            return

        self.snake.insert(0, new_head)

        # Food consumption
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
            self.start_time = time.time()
            self.steps_since_food = 0
        else:
            self.snake.pop()

    def draw(self):
        # Draw game boundary
        pygame.draw.rect(screen, (50, 50, 50), 
                        (self.x_offset, self.y_offset, GAME_WIDTH, GAME_HEIGHT), 1)
        
        # Draw snake
        for segment in self.snake:
            x = self.x_offset + segment[0] * CELL_SIZE
            y = self.y_offset + segment[1] * CELL_SIZE
            color = (0, 255, 0) if self.alive else (100, 100, 100)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
        
        # Draw food
        if self.alive:
            fx = self.x_offset + self.food[0] * CELL_SIZE
            fy = self.y_offset + self.food[1] * CELL_SIZE
            pygame.draw.rect(screen, (255, 0, 0), (fx, fy, CELL_SIZE, CELL_SIZE))

# --- Population Management ---
class Population:
    def __init__(self, size=TOTAL_SIMS):
        self.size = size
        self.networks = [NEATNetwork() for _ in range(size)]
        self.games = []
        self.generation = 1
        self.best_fitness = 0
        self.best_network = None
        self.stagnation_counter = 0
        self.species = defaultdict(list)
        self.load_best_if_exists()
        self.setup_games()

    def save_best(self):
        """Save the best network to file"""
        if self.best_network:
            try:
                save_data = {
                    'network': self.best_network.to_dict(),
                    'generation': self.generation,
                    'fitness': self.best_fitness
                }
                with open('best_snake_genome.json', 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"Saved best genome from generation {self.generation} with fitness {self.best_fitness}")
            except Exception as e:
                print(f"Failed to save: {e}")

    def load_best_if_exists(self):
        """Load the best network if save file exists"""
        try:
            if os.path.exists('best_snake_genome.json'):
                with open('best_snake_genome.json', 'r') as f:
                    save_data = json.load(f)
                
                self.best_network = NEATNetwork.from_dict(save_data['network'])
                self.best_fitness = save_data['fitness']
                saved_gen = save_data['generation']
                
                # Seed population with copies of the best network
                for i in range(min(8, self.size)):  # Seed more copies (8 instead of 5)
                    self.networks[i] = self.best_network.copy()
                
                print(f"Loaded best genome from generation {saved_gen} with fitness {self.best_fitness}")
        except Exception as e:
            print(f"Failed to load saved genome: {e}")

    def setup_games(self):
        self.games = []
        for i, network in enumerate(self.networks):
            row = i // SIMS_PER_ROW
            col = i % SIMS_PER_ROW
            x_offset = col * GAME_WIDTH
            y_offset = row * GAME_HEIGHT
            game = SnakeGame(network, x_offset, y_offset)
            self.games.append(game)

    def step_all(self):
        for game in self.games:
            game.step()

    def all_dead(self):
        return all(not game.alive for game in self.games)

    def calculate_compatibility(self, net1, net2):
        """Calculate how similar two networks are"""
        if not net1.connections or not net2.connections:
            return 0
        
        conn_diff = abs(len(net1.connections) - len(net2.connections))
        node_diff = abs(len(net1.nodes) - len(net2.nodes))
        
        return conn_diff + node_diff

    def speciate(self):
        """Group similar networks into species"""
        self.species = defaultdict(list)
        
        for network in self.networks:
            placed = False
            for species_rep in self.species.keys():
                if self.calculate_compatibility(network, species_rep) < 5:
                    self.species[species_rep].append(network)
                    placed = True
                    break
            
            if not placed:
                self.species[network] = [network]

    def evolve(self):
        # Calculate fitness for all networks
        fitnesses = [game.fitness for game in self.games]
        
        # Track best and check for stagnation
        max_fitness = max(fitnesses) if fitnesses else 0
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            best_idx = fitnesses.index(max_fitness)
            self.best_network = self.networks[best_idx].copy()
            self.stagnation_counter = 0
            self.save_best()  # Auto-save when we find a better network
        else:
            self.stagnation_counter += 1

        # Group networks into species
        self.speciate()
        
        # More aggressive selection for faster evolution
        sorted_pairs = sorted(zip(fitnesses, self.networks), key=lambda x: x[0], reverse=True)
        survivors = max(5, int(self.size * 0.25))  # Slightly reduced survivor rate for more pressure
        
        new_networks = []
        
        # Elite preservation - keep more elites with larger population
        elite_count = max(2, int(self.size * 0.08))  # Increased elite percentage
        for i in range(min(elite_count, len(sorted_pairs))):
            new_networks.append(sorted_pairs[i][1].copy())
        
        # Fill with mutations
        while len(new_networks) < self.size:
            if len(sorted_pairs) > survivors:
                tournament_size = min(5, survivors)
                tournament = random.sample(sorted_pairs[:survivors], tournament_size)
                parent = max(tournament, key=lambda x: x[0])[1]
            else:
                parent = random.choice(sorted_pairs[:survivors])[1]
            
            child = parent.copy()
            child.mutate()
            new_networks.append(child)
        
        self.networks = new_networks
        self.generation += 1
        self.setup_games()

    def draw_best_network(self):
        if not self.best_network:
            return
            
        # Draw best network on the right side
        net_x = SIMS_PER_ROW * GAME_WIDTH + 20
        net_y = 50
        
        input_nodes = [n for n in self.best_network.nodes.values() if n.node_type == 'input']
        hidden_nodes = [n for n in self.best_network.nodes.values() if n.node_type == 'hidden']
        output_nodes = [n for n in self.best_network.nodes.values() if n.node_type == 'output']
        
        node_positions = {}
        
        # Position nodes
        for i, node in enumerate(input_nodes):
            node_positions[node.node_id] = (net_x, net_y + i * 25)  # Tighter spacing
        
        for i, node in enumerate(hidden_nodes):
            node_positions[node.node_id] = (net_x + 100, net_y + i * 30)
        
        for i, node in enumerate(output_nodes):
            node_positions[node.node_id] = (net_x + 200, net_y + i * 40)

        # Draw connections
        for conn in self.best_network.connections:
            if not conn.enabled or conn.input_node not in node_positions or conn.output_node not in node_positions:
                continue
            
            start_pos = node_positions[conn.input_node]
            end_pos = node_positions[conn.output_node]
            
            weight = max(-2, min(2, conn.weight))
            if weight > 0:
                color = (0, int(min(255, abs(weight) * 127)), 0)
            else:
                color = (int(min(255, abs(weight) * 127)), 0, 0)
            
            thickness = max(1, int(abs(weight) * 2))
            pygame.draw.line(screen, color, start_pos, end_pos, thickness)

        # Draw nodes
        for node_id, pos in node_positions.items():
            node = self.best_network.nodes[node_id]
            
            if node.node_type == 'input':
                color = (0, 0, 255)
            elif node.node_type == 'output':
                activation = int(node.value * 255)
                color = (activation, 0, 0)
            else:
                activation = int(node.value * 255)
                color = (activation, activation, 0)
            
            pygame.draw.circle(screen, color, pos, 8)
            pygame.draw.circle(screen, (255, 255, 255), pos, 8, 1)

        # Input labels
        input_labels = ['L_Wall', 'R_Wall', 'T_Wall', 'B_Wall', 
                       'Up_Hit', 'Down_Hit', 'Left_Hit', 'Right_Hit', 
                       'Food_X', 'Food_Y']
        for i, (node_id, pos) in enumerate([(n.node_id, node_positions[n.node_id]) 
                                           for n in input_nodes]):
            if i < len(input_labels):
                label = font.render(input_labels[i], True, (255, 255, 255))
                screen.blit(label, (pos[0] - 60, pos[1] - 5))

        # Output labels
        output_labels = ['Up', 'Down', 'Left', 'Right']
        for i, (node_id, pos) in enumerate([(n.node_id, node_positions[n.node_id]) 
                                           for n in output_nodes]):
            if i < len(output_labels):
                label = font.render(output_labels[i], True, (255, 255, 255))
                screen.blit(label, (pos[0] + 15, pos[1] - 5))

        # Enhanced statistics
        avg_fitness = sum(game.fitness for game in self.games) / len(self.games)
        alive_count = sum(1 for g in self.games if g.alive)
        species_count = len(self.species)
        
        stats = [
            f"Generation: {self.generation}",
            f"Best Fitness: {int(self.best_fitness)}",
            f"Avg Fitness: {int(avg_fitness)}",
            f"Stagnation: {self.stagnation_counter}",
            f"Connections: {len([c for c in self.best_network.connections if c.enabled])}",
            f"Hidden Nodes: {len(hidden_nodes)}",
            f"Species: {species_count}",
            f"Alive: {alive_count}",
            "",
            "Controls:",
            "SPACE - Force evolve",
            "S - Save best",
            "L - Load best"
        ]
        
        for i, stat in enumerate(stats):
            color = (255, 255, 255) if stat else (100, 100, 100)
            text = font.render(stat, True, color)
            screen.blit(text, (net_x, net_y + 300 + i * 20))

    def draw_all(self):
        screen.fill((0, 0, 0))
        
        # Draw all games
        for game in self.games:
            game.draw()
        
        # Draw best network
        self.draw_best_network()

# --- Main Loop ---
population = Population()

running = True
frame_count = 0
simulation_speed = 1  # Steps per frame

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Force evolution
                population.evolve()
            elif event.key == pygame.K_s:
                # Manual save
                population.save_best()
            elif event.key == pygame.K_l:
                # Manual load
                population.load_best_if_exists()
                population.setup_games()
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                # Speed up
                simulation_speed = min(10, simulation_speed + 1)
                print(f"Simulation speed: {simulation_speed}")
            elif event.key == pygame.K_MINUS:
                # Slow down
                simulation_speed = max(1, simulation_speed - 1)
                print(f"Simulation speed: {simulation_speed}")

    # Step simulation multiple times per frame for speed
    for _ in range(simulation_speed):
        population.step_all()
        
        # Check if generation is complete
        if population.all_dead():
            population.evolve()
            break
    
    # Draw everything
    population.draw_all()
    pygame.display.flip()
    clock.tick(FPS)  # Increased to 60 FPS for smoother visualization

pygame.quit()