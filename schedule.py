# schedule.py

# Convert either numeric time (0-5) or string time to standard hour format.
def time_to_hour(time):
    if isinstance(time, int):
        # If it's already numeric (0-5), convert to hour
        time_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
        return time_map.get(time, 0)
    elif isinstance(time, str):
        # If it's a string format, convert to hour
        time_mapping = {
            '10 AM': 10, '11 AM': 11, '12 PM': 12,
            '1 PM': 13, '2 PM': 14, '3 PM': 15
        }
        return time_mapping.get(time, 0)
    return 0

# Assigns activities to rooms, times, facilitators, enrollments, preferences, etc.
class ActivityAssignment:
    def __init__(self, activity, room, time, facilitator):
        self.name = activity.name
        self.room = room
        self.time = time  # Keep as numeric (0-5)
        self.facilitator = facilitator
        self.expected_enrollment = activity.expected_enrollment
        self.preferred_facilitators = activity.preferred_facilitators
        self.other_facilitators = activity.other_facilitators

    # Converts numeric time to string for print display
    def __repr__(self):
        time_map = {0: '10 AM', 1: '11 AM', 2: '12 PM',
                   3: '1 PM', 4: '2 PM', 5: '3 PM'}
        time_str = time_map.get(self.time, 'Unknown Time')
        return (f"Activity: {self.name}, Room: {self.room.name}, "
                f"Time: {time_str}, Facilitator: {self.facilitator.name}")

# Schedule class
class Schedule:
    def __init__(self):
        self.activities = []
        self.fitness = 0

    # Adds an activity to the schedule
    def add_activity(self, activity, room, time, facilitator):
        self.activities.append(ActivityAssignment(activity, room, time, facilitator))

    # Updated using ChatGPT - combined previous implementation and requirements from instructions
    def calculate_fitness(self):
        """Calculate the fitness of the schedule based on the given constraints."""
        fitness = 0
        room_time_combinations = {}
        facilitator_time_slots = {}  # Track facilitator assignments per time slot
        facilitator_total = {}  # Track total activities per facilitator

        # First pass: Process each activity
        for activity in self.activities:
            # Start with 0 for each activity
            activity_fitness = 0

            # Check room-time conflicts (-0.5)
            room_time_key = (activity.room.name, activity.time)
            if room_time_key in room_time_combinations:
                activity_fitness -= 0.5
            room_time_combinations[room_time_key] = True

            # Room size checks
            if activity.room.capacity < activity.expected_enrollment:
                activity_fitness -= 0.5  # Too small
            elif activity.room.capacity > activity.expected_enrollment * 6:
                activity_fitness -= 0.4  # > 6 times
            elif activity.room.capacity > activity.expected_enrollment * 3:
                activity_fitness -= 0.2  # > 3 times
            else:
                activity_fitness += 0.3  # Proper size

            # Facilitator preference checks
            if activity.facilitator.name in activity.preferred_facilitators:
                activity_fitness += 0.5  # Preferred
            elif activity.facilitator.name in activity.other_facilitators:
                activity_fitness += 0.2  # Other listed
            else:
                activity_fitness -= 0.1  # Not listed

            # Track facilitator assignments
            if activity.facilitator.name not in facilitator_time_slots:
                facilitator_time_slots[activity.facilitator.name] = {}
            if activity.time not in facilitator_time_slots[activity.facilitator.name]:
                facilitator_time_slots[activity.facilitator.name][activity.time] = []
            facilitator_time_slots[activity.facilitator.name][activity.time].append(activity)

            # Track total activities per facilitator
            facilitator_total[activity.facilitator.name] = facilitator_total.get(activity.facilitator.name, 0) + 1

            fitness += activity_fitness

        # Second pass: Facilitator load checks
        for facilitator, time_slots in facilitator_time_slots.items():
            # Check activities per time slot
            for time_slot, activities in time_slots.items():
                if len(activities) == 1:
                    fitness += 0.2  # Single activity bonus
                else:
                    fitness -= 0.2 * (len(activities) - 1)  # Multiple activities penalty

            # Check total activities
            total_activities = facilitator_total[facilitator]
            if total_activities > 4:
                fitness -= 0.5  # Overload penalty
            elif total_activities <= 2 and facilitator != "Tyler":
                fitness -= 0.4  # Underload penalty

            # Check consecutive time slots
            times = sorted(time_slots.keys())
            for i in range(len(times) - 1):
                if times[i + 1] - times[i] == 1:
                    # Handle like SLA191/101 consecutive rules
                    for act1 in time_slots[times[i]]:
                        for act2 in time_slots[times[i + 1]]:
                            # Apply building separation rules if applicable
                            if any(act.name.startswith(("SLA191", "SLA101")) for act in [act1, act2]):
                                in_specific_building = lambda r: r.name.startswith(("Roman", "Beach"))
                                if in_specific_building(act1.room) != in_specific_building(act2.room):
                                    fitness -= 0.4

        # Third pass: SLA101 and SLA191 specific rules
        sla101_sections = [a for a in self.activities if a.name.startswith("SLA101")]
        sla191_sections = [a for a in self.activities if a.name.startswith("SLA191")]

        # SLA101 rules
        if len(sla101_sections) == 2:
            time_diff = abs(sla101_sections[0].time - sla101_sections[1].time)
            if time_diff > 4:
                fitness += 0.5  # More than 4 hours apart
            elif time_diff == 0:
                fitness -= 0.5  # Same time slot

        # SLA191 rules
        if len(sla191_sections) == 2:
            time_diff = abs(sla191_sections[0].time - sla191_sections[1].time)
            if time_diff > 4:
                fitness += 0.5  # More than 4 hours apart
            elif time_diff == 0:
                fitness -= 0.5  # Same time slot

        # SLA101 and SLA191 relationship rules
        for sla101 in sla101_sections:
            for sla191 in sla191_sections:
                time_diff = abs(sla101.time - sla191.time)
                if time_diff == 1:  # Consecutive slots
                    fitness += 0.5
                    # Check building separation only for consecutive slots
                    in_specific_building = lambda r: r.name.startswith(("Roman", "Beach"))
                    if in_specific_building(sla101.room) != in_specific_building(sla191.room):
                        fitness -= 0.4
                elif time_diff == 2:  # One hour separation
                    fitness += 0.25
                elif time_diff == 0:  # Same time slot
                    fitness -= 0.25

        self.fitness = fitness
        return fitness