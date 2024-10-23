# room.py

class Room:
    """Represents a room that an activity can be scheduled in."""
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity

    def __repr__(self):
        return f"{self.name} (Capacity: {self.capacity})"
