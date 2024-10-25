# schedule.py

def time_to_hour(time_str):
    """Converts a time string (like '10 AM') to an integer representing the hour (like 10)."""
    time_mapping = {
        '10 AM': 10, '11 AM': 11, '12 PM': 12, '1 PM': 13, '2 PM': 14, '3 PM': 15
    }
    return time_mapping.get(time_str, 0)


class ActivityAssignment:
    def __init__(self, activity, room, time, facilitator):
        self.name = activity.name
        self.room = room
        self.time = time
        self.facilitator = facilitator
        self.expected_enrollment = activity.expected_enrollment
        self.preferred_facilitators = activity.preferred_facilitators
        self.other_facilitators = activity.other_facilitators

    def __repr__(self):
        return (f"Activity: {self.name}, Room: {self.room.name}, "
                f"Time: {self.time}, Facilitator: {self.facilitator.name}")


class Schedule:
    def __init__(self):
        self.activities = []
        self.fitness = 0

    def add_activity(self, activity, room, time, facilitator):
        """Add an activity to the schedule."""
        self.activities.append(ActivityAssignment(activity, room, time, facilitator))

    def calculate_fitness(self):
        """Calculate the fitness of the schedule based on the given constraints."""
        fitness = 0

        # Track room-time combinations to penalize activities scheduled in the same room and time
        room_time_combinations = {}

        # Step 1: Room size penalties and room-time conflicts
        for activity in self.activities:
            # Check for same time, same room conflicts
            room_time_key = (activity.room.name, activity.time)
            if room_time_key not in room_time_combinations:
                room_time_combinations[room_time_key] = 1
            else:
                room_time_combinations[room_time_key] += 1
                fitness -= 0.5  # Penalty for room-time conflict

            # Room size penalties
            if activity.room.capacity < activity.expected_enrollment:
                fitness -= 0.5  # Room too small
            elif activity.room.capacity > activity.expected_enrollment * 6:
                fitness -= 0.4  # Room more than 6x enrollment
            elif activity.room.capacity > activity.expected_enrollment * 3:
                fitness -= 0.2  # Room more than 3x enrollment
            else:
                fitness += 0.3  # Proper room size

            # Step 2: Facilitator assignment
            if activity.facilitator.name in activity.preferred_facilitators:
                fitness += 0.5  # Preferred facilitator
            elif activity.facilitator.name in activity.other_facilitators:
                fitness += 0.2  # Acceptable facilitator
            else:
                fitness -= 0.1  # Not preferred facilitator

        # Step 3: Facilitator load penalties
        facilitator_schedule = {}
        for activity in self.activities:
            if activity.facilitator not in facilitator_schedule:
                facilitator_schedule[activity.facilitator] = []
            facilitator_schedule[activity.facilitator].append(activity.time)

        # Step 3: Facilitator load penalties
        for facilitator, times in facilitator_schedule.items():
            # Check if facilitator is overseeing more than 4 activities
            if len(times) > 4:
                fitness -= 0.5  # Too many activities

            # Check for scheduling conflicts (multiple activities at the same time)
            for time in times:
                if times.count(time) > 1:
                    fitness -= 0.2  # Multiple activities in the same time slot

            # Check if facilitator is overseeing only 1 or 2 activities (except for Tyler)
            if len(times) <= 2 and facilitator.name != "Tyler":
                fitness -= 0.4

        # Step 4: SLA 101 and SLA 191 adjustments
        sla101_times = [time_to_hour(a.time) for a in self.activities if a.name.startswith("SLA101")]
        sla191_times = [time_to_hour(a.time) for a in self.activities if a.name.startswith("SLA191")]

        # SLA 101 conditions
        if sla101_times and max(sla101_times) - min(sla101_times) > 4:
            fitness += 0.5  # Sections of SLA 101 more than 4 hours apart
        if len(set(sla101_times)) < 2:
            fitness -= 0.5  # Both sections in the same time slot

        # SLA 191 conditions
        if sla191_times and max(sla191_times) - min(sla191_times) > 4:
            fitness += 0.5  # Sections of SLA 191 more than 4 hours apart
        if len(set(sla191_times)) < 2:
            fitness -= 0.5  # Both sections in the same time slot

        # Cross-over penalties/bonuses between SLA 101 and SLA 191
        if any(abs(t1 - t2) == 1 for t1 in sla101_times for t2 in sla191_times):
            fitness += 0.5  # SLA 101 and SLA 191 are in consecutive time slots
        if any(abs(t1 - t2) == 1 for t1 in sla101_times for t2 in sla191_times) and (
                any(a.room.name in ["Roman", "Beach"] for a in self.activities if a.name.startswith("SLA101")) ^
                any(a.room.name in ["Roman", "Beach"] for a in self.activities if a.name.startswith("SLA191"))
        ):
            fitness -= 0.4  # SLA 101 and SLA 191 consecutive but in widely separated rooms

        self.fitness = fitness
