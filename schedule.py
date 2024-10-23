# schedule.py

class Schedule:
    """Represents a full schedule of activities."""
    def __init__(self):
        self.activities = []  # List to store activity assignments
        self.fitness = 0

    def add_activity(self, activity, room, time, facilitator):
        """Adds an activity with room, time, and facilitator to the schedule."""
        activity.room = room
        activity.time = time
        activity.facilitator = facilitator
        self.activities.append(activity)

    def calculate_fitness(self):
        """Calculates the fitness score for the schedule based on the problem's requirements."""
        fitness = 0
        room_time_conflicts = set()

        for activity in self.activities:
            room = activity.room
            time = activity.time
            facilitator = activity.facilitator

            # Check for room-time conflicts
            if (room.name, time) in room_time_conflicts:
                fitness -= 0.5  # Penalty for room conflict
            else:
                room_time_conflicts.add((room.name, time))

            # Check room size
            if room.capacity < activity.expected_enrollment:
                fitness -= 0.5  # Penalty for room too small
            elif room.capacity > activity.expected_enrollment * 3:
                if room.capacity > activity.expected_enrollment * 6:
                    fitness -= 0.4  # Penalty for room too large (>6 times enrollment)
                else:
                    fitness -= 0.2  # Penalty for room too large (>3 times enrollment)
            else:
                fitness += 0.3  # Reward for appropriate room size

            # Check facilitator assignment
            if facilitator in activity.preferred_facilitators:
                fitness += 0.5  # Preferred facilitator
            elif facilitator in activity.other_facilitators:
                fitness += 0.2  # Other acceptable facilitator
            else:
                fitness -= 0.1  # Not preferred or acceptable facilitator

        self.fitness = fitness
        return fitness
