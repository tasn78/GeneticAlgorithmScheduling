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
        last_improvement_gen = 0
        current_mutation_rate = self.mutation_rate  # Start with initial mutation rate of 0.01
        baseline_fitness = None  # Baseline to track fitness improvements

        for generation in range(self.generations):
            print(f"Generation {generation}")

            # Sort the population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness = self.population[0].fitness
            average_fitness = sum([s.fitness for s in self.population]) / self.population_size

            best_fitness_over_time.append(best_fitness)
            average_fitness_over_time.append(average_fitness)

            # Set initial baseline fitness in the first generation
            if generation == 0:
                baseline_fitness = best_fitness

            # Check for improvement beyond the baseline fitness
            if best_fitness > baseline_fitness:
                # If there is an improvement, halve the mutation rate
                current_mutation_rate /= 2
                print(f"Improvement found. Adjusting mutation rate to: {current_mutation_rate}")
                baseline_fitness = best_fitness  # Update the baseline to the new best fitness

            # Track if there has been a significant improvement
            if generation > 0 and (best_fitness - best_fitness_over_time[last_improvement_gen]) >= 0.01 * best_fitness:
                last_improvement_gen = generation

            # Early stopping condition
            if generation - last_improvement_gen > 100:
                print(f"Stopping early at generation {generation} due to no significant improvement.")
                break

            # Elitism: preserve the top-performing individuals
            elitism_size = int(self.elitism_ratio * self.population_size)
            new_population = self.population[:elitism_size]

            # Generate new population through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)

                # Apply mutation with the current mutation rate
                if random.random() < current_mutation_rate:
                    self.mutate(offspring1)
                if random.random() < current_mutation_rate:
                    self.mutate(offspring2)

                offspring1.calculate_fitness()
                offspring2.calculate_fitness()

                new_population.append(offspring1)
                new_population.append(offspring2)

            # Replace the current population with the new one
            self.population = new_population[:self.population_size]

        # Plotting fitness over time
        plt.plot(best_fitness_over_time, label="Best Fitness")
        plt.plot(average_fitness_over_time, label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Progress Over Generations")
        plt.legend()
        plt.show()

        # Print and save the best schedule
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
        """Perform intelligent crossover between two schedules."""
        offspring1, offspring2 = Schedule(), Schedule()

        # Track processed activities to avoid duplicates
        processed_activities = set()

        # Keep track of used rooms and times for conflict avoidance
        used_slots1 = {}  # For offspring1
        used_slots2 = {}  # For offspring2

        def is_valid_assignment(used_slots, room, time):
            return (room.name, time) not in used_slots

        # Helper function to add activity while tracking slots
        def add_activity_safely(schedule, used_slots, act_assignment):
            if is_valid_assignment(used_slots, act_assignment.room, act_assignment.time):
                orig_activity = next(act for act in self.activities
                                     if act.name == act_assignment.name)
                schedule.add_activity(orig_activity,
                                      act_assignment.room,
                                      act_assignment.time,
                                      act_assignment.facilitator)
                used_slots[(act_assignment.room.name, act_assignment.time)] = True
                processed_activities.add(act_assignment.name)
                return True
            return False

        # First handle SLA100 sections
        sla100_activities1 = [act for act in parent1.activities
                              if act.name.startswith('SLA100')]
        sla100_activities2 = [act for act in parent2.activities
                              if act.name.startswith('SLA100')]

        if random.random() < 0.5:
            for act in sla100_activities1:
                add_activity_safely(offspring1, used_slots1, act)
            for act in sla100_activities2:
                add_activity_safely(offspring2, used_slots2, act)
        else:
            for act in sla100_activities2:
                add_activity_safely(offspring1, used_slots1, act)
            for act in sla100_activities1:
                add_activity_safely(offspring2, used_slots2, act)

        # Then handle SLA191 sections
        sla191_activities1 = [act for act in parent1.activities
                              if act.name.startswith('SLA191')]
        sla191_activities2 = [act for act in parent2.activities
                              if act.name.startswith('SLA191')]

        if random.random() < 0.5:
            for act in sla191_activities1:
                add_activity_safely(offspring1, used_slots1, act)
            for act in sla191_activities2:
                add_activity_safely(offspring2, used_slots2, act)
        else:
            for act in sla191_activities2:
                add_activity_safely(offspring1, used_slots1, act)
            for act in sla191_activities1:
                add_activity_safely(offspring2, used_slots2, act)

        # Handle remaining activities
        remaining_activities1 = [act for act in parent1.activities
                                 if act.name not in processed_activities]
        remaining_activities2 = [act for act in parent2.activities
                                 if act.name not in processed_activities]

        # Ensure we have activities to process
        for act1, act2 in zip(remaining_activities1, remaining_activities2):
            if random.random() < 0.5:
                add_activity_safely(offspring1, used_slots1, act1)
                add_activity_safely(offspring2, used_slots2, act2)
            else:
                add_activity_safely(offspring1, used_slots1, act2)
                add_activity_safely(offspring2, used_slots2, act1)

        # Ensure all activities are scheduled
        all_activities = set(act.name for act in self.activities)
        scheduled_activities1 = set(act.name for act in offspring1.activities)
        scheduled_activities2 = set(act.name for act in offspring2.activities)

        # Add any missing activities with random assignments
        for activity_name in all_activities - scheduled_activities1:
            activity = next(act for act in self.activities if act.name == activity_name)
            while True:
                random_room = random.choice(self.rooms)
                random_time = random.randint(0, 5)
                if (random_room.name, random_time) not in used_slots1:
                    offspring1.add_activity(activity, random_room, random_time,
                                            random.choice(self.facilitators))
                    used_slots1[(random_room.name, random_time)] = True
                    break

        for activity_name in all_activities - scheduled_activities2:
            activity = next(act for act in self.activities if act.name == activity_name)
            while True:
                random_room = random.choice(self.rooms)
                random_time = random.randint(0, 5)
                if (random_room.name, random_time) not in used_slots2:
                    offspring2.add_activity(activity, random_room, random_time,
                                            random.choice(self.facilitators))
                    used_slots2[(random_room.name, random_time)] = True
                    break

        return offspring1, offspring2

    def mutate(self, schedule):
        """Intelligent mutation that respects constraints."""
        mutation_occurred = False

        # Track used room-time combinations
        used_slots = {(act.room.name, act.time) for act in schedule.activities}

        for activity in schedule.activities:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(["room", "time", "facilitator"])

                if mutation_type == "room":
                    # Choose rooms that can accommodate the activity
                    suitable_rooms = [room for room in self.rooms
                                      if room.capacity >= activity.expected_enrollment]
                    if suitable_rooms:
                        new_room = random.choice(suitable_rooms)
                        if (new_room.name, activity.time) not in used_slots:
                            used_slots.remove((activity.room.name, activity.time))
                            activity.room = new_room
                            used_slots.add((new_room.name, activity.time))
                            mutation_occurred = True

                elif mutation_type == "time":
                    # Special handling for SLA100 and SLA191 sections
                    base_name = activity.name[:6]
                    if base_name in ['SLA100', 'SLA191']:
                        # Find the other section's time
                        other_section = next((act for act in schedule.activities
                                              if act.name.startswith(base_name) and act != activity), None)
                        if other_section:
                            # Try to maintain good spacing
                            possible_times = [t for t in range(6)
                                              if abs(t - other_section.time) > 2
                                              and (activity.room.name, t) not in used_slots]
                        else:
                            possible_times = [t for t in range(6)
                                              if (activity.room.name, t) not in used_slots]
                    else:
                        possible_times = [t for t in range(6)
                                          if (activity.room.name, t) not in used_slots]

                    if possible_times:
                        used_slots.remove((activity.room.name, activity.time))
                        activity.time = random.choice(possible_times)
                        used_slots.add((activity.room.name, activity.time))
                        mutation_occurred = True

                elif mutation_type == "facilitator":
                    # Prefer facilitators from preferred or other lists
                    weighted_facilitators = (
                            [(f, 3) for f in self.facilitators if f.name in activity.preferred_facilitators] +
                            [(f, 2) for f in self.facilitators if f.name in activity.other_facilitators] +
                            [(f, 1) for f in self.facilitators]
                    )
                    if weighted_facilitators:
                        facilitators, weights = zip(*weighted_facilitators)
                        activity.facilitator = random.choices(facilitators, weights=weights, k=1)[0]
                        mutation_occurred = True

        if mutation_occurred:
            schedule.calculate_fitness()

        return mutation_occurred

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

