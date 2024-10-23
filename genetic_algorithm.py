# genetic_algorithm.py
import random
import numpy as np
from schedule import Schedule

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, activities, rooms, times, facilitators):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.activities = activities
        self.rooms = rooms
        self.times = times
        self.facilitators = facilitators
        self.population = []
        self.best_schedule = None  # Track the best schedule
        self.best_fitness = -float('inf')  # Track the best fitness

    def initialize_population(self):
        """Initialize the population with random schedules."""
        self.population = []
        for _ in range(self.population_size):
            schedule = Schedule()
            for activity in self.activities:
                # Random room, time, and facilitator
                random_room = random.choice(self.rooms)
                random_time = random.choice(self.times)
                random_facilitator = random.choice(self.facilitators)
                schedule.add_activity(activity, random_room, random_time, random_facilitator)
            self.population.append(schedule)

    def evolve_population(self):
        """Evolve the population through generations with dynamic mutation rate adjustment."""
        best_fitness_over_time = []
        for generation in range(self.generations):
            print(f"Generation {generation}")
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            current_best = self.population[0]
            current_best_fitness = current_best.fitness

            # Keep track of the best schedule and fitness
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_schedule = current_best

            best_fitness_over_time.append(current_best_fitness)

            if generation > 100 and best_fitness_over_time[-1] <= best_fitness_over_time[-101] * 1.01:
                # If improvement in fitness is less than 1% over 100 generations, halve mutation rate
                self.mutation_rate /= 2

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                self.mutate(offspring1)
                self.mutate(offspring2)
                offspring1.calculate_fitness()
                offspring2.calculate_fitness()
                new_population.append(offspring1)
                new_population.append(offspring2)
            self.population = new_population

        self.print_best_schedule()

    def softmax(self, fitness_values):
        """Apply softmax normalization to fitness values."""
        exp_values = np.exp(fitness_values - np.max(fitness_values))
        return exp_values / np.sum(exp_values)

    def select_parents(self):
        """Select two parents based on softmax-normalized fitness."""
        fitness_values = [sched.fitness for sched in self.population]
        probabilities = self.softmax(fitness_values)
        return random.choices(self.population, k=2, weights=probabilities)

    def crossover(self, parent1, parent2):
        """Perform crossover between two schedules."""
        offspring1, offspring2 = Schedule(), Schedule()

        # Mix parent schedules while trying to avoid room-time conflicts
        for i, (activity1, activity2) in enumerate(zip(parent1.activities, parent2.activities)):
            if random.random() > 0.5:
                # Inherit from parent 1
                offspring1.activities.append(activity1)
                offspring2.activities.append(activity2)
            else:
                # Inherit from parent 2
                offspring1.activities.append(activity2)
                offspring2.activities.append(activity1)

        return offspring1, offspring2

    def mutate(self, schedule):
        """Mutate a schedule by randomly altering rooms, times, or facilitators."""
        for activity in schedule.activities:
            if random.random() < self.mutation_rate:
                # Randomly change either room, time, or facilitator for more variation
                change_type = random.choice(["room", "time", "facilitator"])
                if change_type == "room":
                    new_room = random.choice(self.rooms)
                    activity.room = new_room
                elif change_type == "time":
                    new_time = random.choice(self.times)
                    activity.time = new_time
                elif change_type == "facilitator":
                    new_facilitator = random.choice(self.facilitators)
                    activity.facilitator = new_facilitator

    def print_best_schedule(self):
        """Print and output the best schedule to a file."""
        if self.best_schedule:
            print(f"\nBest Schedule Fitness: {self.best_fitness}")
            with open('best_schedule.txt', 'w') as f:
                for activity in self.best_schedule.activities:
                    result = f"Activity: {activity.name}, Room: {activity.room.name}, Time: {activity.time}, Facilitator: {activity.facilitator.name}"
                    print(result)
                    f.write(result + "\n")
            print("\nBest schedule printed to 'best_schedule.txt'.")
