# activity.py

class Activity:
    """Represents an activity to be scheduled."""
    def __init__(self, name, expected_enrollment, preferred_facilitators, other_facilitators):
        self.name = name
        self.expected_enrollment = expected_enrollment
        self.preferred_facilitators = preferred_facilitators
        self.other_facilitators = other_facilitators
        self.room = None
        self.time = None
        self.facilitator = None

    def __repr__(self):
        return f"{self.name} - Room: {self.room.name}, Time: {self.time}, Facilitator: {self.facilitator}"
