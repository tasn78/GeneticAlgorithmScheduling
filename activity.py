# activity.py

class Activity:
    """Represents an activity to be scheduled."""
    def __init__(self, name, expected_enrollment, preferred_facilitators, other_facilitators):
        self.name = name
        self.expected_enrollment = expected_enrollment
        self.preferred_facilitators = preferred_facilitators  # Preferred facilitators for the activity
        self.other_facilitators = other_facilitators  # Facilitators that can also handle the activity
