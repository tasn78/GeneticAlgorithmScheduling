# genetic_algorithm.py
import random
import numpy as np
from schedule import Schedule
import matplotlib.pyplot as plt  # Importing for fitness plotting


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, activities, rooms, times, facilitators, temperature=2.0, elitism_ratio=0.05):
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
        self.temperature = temperature  # Softmax temperature
        self.elitism_ratio = elitism_ratio  # Elitism ratio (e.g., 5%)

    def initialize_population(self):
        """Initialize the population with random schedules."""
        self.population = []
        for _ in range(self.population_size):
            schedule = Schedule()
            for activity in self.activities:
                # Randomly assign room, time, and facilitator
                random_room = random.choice(self.rooms)
                random_time = random.randint(0, 5)  # Use numeric time (0 = 10 AM, ..., 5 = 3 PM)
                random_facilitator = random.choice(self.facilitators)
                schedule.add_activity(activity, random_room, random_time, random_facilitator)
            schedule.calculate_fitness()  # Calculate fitness immediately after initialization
            self.population.append(schedule)

    def evolve_population(self):
        """Evolve the population through generations based on fitness with elitism and temperature scaling."""
        best_fitness_over_time = []
        average_fitness_over_time = []
        last_improvement_gen = 0  # Track the last generation with significant improvement

        for generation in range(self.generations):
            print(f"Generation {generation}")

            # Sort the population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness = self.population[0].fitness
            average_fitness = sum([s.fitness for s in self.population]) / self.population_size

            best_fitness_over_time.append(best_fitness)
            average_fitness_over_time.append(average_fitness)

            # Track if there has been a significant improvement
            if generation > 0 and (best_fitness - best_fitness_over_time[last_improvement_gen]) >= 0.01 * best_fitness:
                last_improvement_gen = generation  # Update last improvement generation

            # Early stopping condition if no significant improvement over 100 generations
            if generation - last_improvement_gen > 100:
                print(f"Stopping early at generation {generation} due to no significant improvement.")
                break

            # Elitism: Preserve the top 5% of the population
            elitism_size = int(self.elitism_ratio * self.population_size)
            new_population = self.population[:elitism_size]

            # Generate new population through crossover and mutation for the rest
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()  # Softmax-based parent selection with temperature
                offspring1, offspring2 = self.crossover(parent1, parent2)

                # Apply mutation and recalculate fitness
                self.mutate(offspring1)
                self.mutate(offspring2)
                offspring1.calculate_fitness()
                offspring2.calculate_fitness()

                # Add to new population
                new_population.append(offspring1)
                new_population.append(offspring2)

            # Replace the current population with the new one
            self.population = new_population[:self.population_size]  # Ensure population size stays constant

        # Plotting fitness over time
        plt.plot(best_fitness_over_time, label="Best Fitness")
        plt.plot(average_fitness_over_time, label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Progress Over Generations")
        plt.legend()
        plt.show()

        # Print and save the best schedule after evolution
        best_schedule = self.population[0]
        self.print_best_schedule(best_schedule, generation)

    def softmax(self, fitness_values, temperature=1.0):
        """Apply softmax normalization to fitness values with temperature scaling."""
        scaled_values = np.array(fitness_values) / temperature  # Scale by temperature
        exp_values = np.exp(scaled_values - np.max(scaled_values))  # Shift for numerical stability
        return exp_values / np.sum(exp_values)  # Return normalized probabilities

    def select_parents(self):
        """Select two parents based on softmax-normalized fitness with temperature scaling."""
        fitness_values = [sched.fitness for sched in self.population]
        probabilities = self.softmax(fitness_values, temperature=self.temperature)  # Use temperature scaling
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
        mutation_occurred = False  # Track if mutation occurred
        for activity in schedule.activities:
            if random.random() < self.mutation_rate:
                change_type = random.choice(["room", "time", "facilitator"])

                if change_type == "room":
                    new_room = random.choice(self.rooms)
                    activity.room = new_room

                elif change_type == "time":
                    new_time = random.randint(0, 5)  # Mutate time as numeric value (0 = 10 AM, ..., 5 = 3 PM)
                    activity.time = new_time

                elif change_type == "facilitator":
                    new_facilitator = random.choice(self.facilitators)
                    activity.facilitator = new_facilitator

                schedule.calculate_fitness()  # Recalculate fitness after mutation
                mutation_occurred = True

        return mutation_occurred  # Return if mutation occurred

    def print_best_schedule(self, best_schedule, generation):
        """Print the best schedule to a file and console, including the generation number."""
        with open('best_schedule.txt', 'w') as file:
            file.write(f"Best Schedule Fitness: {best_schedule.fitness}\n")
            file.write(f"Best Schedule found at Generation: {generation}\n")  # Add generation number
            for activity in best_schedule.activities:
                time_string = self.convert_time_to_string(activity.time)  # Convert time back to string
                file.write(
                    f"Activity: {activity.name}, Room: {activity.room.name}, Time: {time_string}, Facilitator: {activity.facilitator}\n")
                print(
                    f"Activity: {activity.name}, Room: {activity.room.name}, Time: {time_string}, Facilitator: {activity.facilitator}")

    def convert_time_to_string(self, time_numeric):
        """Convert numeric time back to a string."""
        time_map = {0: '10 AM', 1: '11 AM', 2: '12 PM', 3: '1 PM', 4: '2 PM', 5: '3 PM'}
        return time_map.get(time_numeric, 'Unknown Time')
