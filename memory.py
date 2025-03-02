import json
import os
import datetime

class MemoryManager:
    def __init__(self, memory_file="memory.json"):
        self.memory_file = memory_file
        self.memory = {"short_term": [], "long_term": {}}
        self.load_memory()

    def load_memory(self):
        """Loads memory from a file (persistent storage)."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        else:
            self.save_memory()

    def save_memory(self):
        """Saves memory to a file."""
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def store_short_term(self, data):
        """Stores temporary information (recent conversations, actions)."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.memory["short_term"].append({"time": timestamp, "data": data})

        # Keep only the last 10 short-term memories
        if len(self.memory["short_term"]) > 10:
            self.memory["short_term"].pop(0)
        
        self.save_memory()

    def store_long_term(self, key, value):
        """Stores long-term memory (preferences, user habits, etc.)."""
        self.memory["long_term"][key] = value
        self.save_memory()

    def recall_memory(self, key=None):
        """Retrieves memory; if key is provided, fetches specific info."""
        if key:
            return self.memory["long_term"].get(key, "No record found.")
        return self.memory

    def clear_short_term(self):
        """Clears short-term memory to free up space."""
        self.memory["short_term"] = []
        self.save_memory()

    def clear_memory(self):
        """Resets all memory (USE WITH CAUTION)."""
        self.memory = {"short_term": [], "long_term": {}}
        self.save_memory()

# Example Usage
if __name__ == "__main__":
    mem = MemoryManager()
    mem.store_short_term("User asked about the weather.")
    mem.store_long_term("favorite_color", "Blue")

    print(mem.recall_memory())  # Prints all memory
    print(mem.recall_memory("favorite_color"))  # Prints "Blue"
