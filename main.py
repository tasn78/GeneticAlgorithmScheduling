# main.py
from activity import Activity
from room import Room
from facilitator import Facilitator
from genetic_algorithm import GeneticAlgorithm


def main():
    # Define the activities with their expected enrollment, preferred, and other facilitators
    activities = [
        Activity('SLA100A', 50, ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards']),
        Activity('SLA100B', 50, ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards']),
        Activity('SLA191A', 50, ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards']),
        Activity('SLA191B', 50, ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards']),
        Activity('SLA201', 50, ['Glen', 'Banks', 'Zeldin', 'Shaw'], ['Numen', 'Richards', 'Singer']),
        Activity('SLA291', 50, ['Lock', 'Banks', 'Zeldin', 'Singer'], ['Numen', 'Richards', 'Shaw', 'Tyler']),
        Activity('SLA303', 60, ['Glen', 'Zeldin', 'Banks'], ['Numen', 'Singer', 'Shaw']),
        Activity('SLA304', 25, ['Glen', 'Banks', 'Tyler'], ['Numen', 'Singer', 'Shaw', 'Richards', 'Uther', 'Zeldin']),
        Activity('SLA394', 20, ['Tyler', 'Singer'], ['Richards', 'Zeldin']),
        Activity('SLA449', 60, ['Tyler', 'Singer', 'Shaw'], ['Zeldin', 'Uther']),
        Activity('SLA451', 100, ['Tyler', 'Singer', 'Shaw'], ['Zeldin', 'Uther', 'Richards', 'Banks'])
    ]

    # Define the rooms with their capacities
    rooms = [
        Room('Slater 003', 45),
        Room('Roman 216', 30),
        Room('Loft 206', 75),
        Room('Roman 201', 50),
        Room('Loft 310', 108),
        Room('Beach 201', 60),
        Room('Beach 301', 75),
        Room('Logos 325', 450),
        Room('Frank 119', 60)
    ]

    # Define the available time slots
    times = ['10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM']

    # Define the facilitators
    facilitators = [
        Facilitator('Lock'), Facilitator('Glen'), Facilitator('Banks'), Facilitator('Richards'),
        Facilitator('Shaw'), Facilitator('Singer'), Facilitator('Uther'), Facilitator('Tyler'),
        Facilitator('Numen'), Facilitator('Zeldin')
    ]

    # Initialize Genetic Algorithm
    ga = GeneticAlgorithm(population_size=500, mutation_rate=0.01, generations=1000,
                          activities=activities, rooms=rooms, times=times, facilitators=facilitators)

    ga.initialize_population()

    for generation in range(100):
        print(f"Generation {generation}")
        ga.evolve_population()

    print(f"Best schedule fitness: {ga.population[0].fitness}")
    for activity in ga.population[0].activities:
        print(activity)


if __name__ == '__main__':
    main()