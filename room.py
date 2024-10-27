# room.py

# Rooms for activities to be scheduled in
class Room:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity

    def __repr__(self):
        return f"{self.name} (Capacity: {self.capacity})"
