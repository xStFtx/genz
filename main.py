import numpy as np
import random

# --- NEURAL NETWORK SETUP ---

ACTIVATIONS = {
    'tanh': np.tanh,
    'sigmoid': lambda x: 1 / (1 + np.exp(-x))
}

def create_random_weights(layer_sizes):
    """Initialize weights and biases for the neural network."""
    return [(np.random.randn(*shape), np.random.randn(shape[1])) for shape in zip(layer_sizes[:-1], layer_sizes[1:])]

def forward_pass(x, weights, activation='tanh'):
    """Perform a forward pass through the neural network."""
    act_func = ACTIVATIONS[activation]
    a = x
    for w, b in weights:
        z = np.dot(a, w) + b
        a = act_func(z)
    return a

# --- GENETIC ALGORITHM FUNCTIONS ---

def fitness(weights, data, target):
    """Calculate the fitness of a particular set of weights."""
    predictions = forward_pass(data, weights)
    mse = np.mean((predictions - target) ** 2)
    # Consider the magnitude of the weights in the fitness to promote simpler networks
    weight_magnitude = sum(np.mean(np.abs(w)) for w, _ in weights)
    return 1 / (1 + mse + 0.01 * weight_magnitude)

def uniform_crossover(parent1, parent2):
    """Uniform crossover between two parents."""
    child = []
    for (w1, b1), (w2, b2) in zip(parent1, parent2):
        cw = np.where(np.random.rand(*w1.shape) < 0.5, w1, w2)
        cb = np.where(np.random.rand(*b1.shape) < 0.5, b1, b2)
        child.append((cw, cb))
    return child

def one_point_crossover(parent1, parent2):
    """One-point crossover between two parents."""
    split_point = random.randint(0, len(parent1))
    child = parent1[:split_point] + parent2[split_point:]
    return child

def mutate(weights, mutation_rate):
    """Apply mutations to the weights."""
    return [(w + mutation_rate * np.random.randn(*w.shape), b + mutation_rate * np.random.randn(*b.shape)) for w, b in weights]

def genetic_algorithm(data, target, layer_sizes, generations=100, population_size=100, elitism_rate=0.1, checkpoint_interval=10):
    """Optimize neural network weights using a genetic algorithm."""
    population = [create_random_weights(layer_sizes) for _ in range(population_size)]
    last_best_fitness = 0

    for generation in range(generations):
        fitness_scores = np.array([fitness(weights, data, target) for weights in population])
        sorted_indices = fitness_scores.argsort()[::-1]
        elites = [population[i] for i in sorted_indices[:int(elitism_rate * population_size)]]

        if fitness_scores[sorted_indices[0]] < 0.7 * last_best_fitness:
            handle_catastrophic_forgetting(population, sorted_indices, data, target)

        offspring = produce_offspring(population, population_size, elitism_rate, last_best_fitness, fitness_scores[sorted_indices[0]], data, target)

        population = elites + offspring

        last_best_fitness = fitness_scores[sorted_indices[0]]

    return population[sorted_indices[0]]

def handle_catastrophic_forgetting(population, sorted_indices, data, target):
    """Handle catastrophic forgetting by retraining the best individual."""
    print("Possible catastrophic forgetting detected. Re-training best individual.")
    best_weights = mutate(population[sorted_indices[0]], 0.1)
    population[sorted_indices[0]] = best_weights

def produce_offspring(population, population_size, elitism_rate, last_best_fitness, current_best_fitness, data, target):
    """Produce offspring for the next generation."""
    mutation_rate = 0.1 + 0.1 * (last_best_fitness - current_best_fitness)
    offspring = []
    for _ in range(population_size - int(elitism_rate * population_size)):
        parent1, parent2 = tournament_select(population, data, target)
        crossover_func = random.choice([uniform_crossover, one_point_crossover])
        child = mutate(crossover_func(parent1, parent2), mutation_rate)
        offspring.append(child)
    return offspring

def tournament_select(population, data, target):
    """Select two parents using tournament selection."""
    tournament = random.sample(population, 5)
    sorted_parents = sorted(tournament, key=lambda x: fitness(x, data, target), reverse=True)
    return sorted_parents[0], sorted_parents[1]

# --- TEST ---

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])
layer_sizes = [2, 10, 10, 1]  # Two hidden layers with 10 neurons each
best_weights = genetic_algorithm(data, target, layer_sizes)

for d, t in zip(data, target):
    prediction = forward_pass(d, best_weights)
    print(f"Input: {d}, Prediction: {prediction[0]:.2f}, Actual: {t[0]}")

