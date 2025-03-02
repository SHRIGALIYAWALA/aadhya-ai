import json
import os

class DigitalTwin:
    def __init__(self, user_profile_path="features/data/user_profile.json"):
        self.user_profile_path = user_profile_path
        self.user_data = self.load_user_profile()

    def load_user_profile(self):
        """Loads the user's digital twin profile from a JSON file."""
        if os.path.exists(self.user_profile_path):
            with open(self.user_profile_path, "r") as file:
                return json.load(file)
        return {"name": "User", "preferences": {}, "behavior_patterns": {}}

    def update_user_profile(self, key, value):
        """Updates and saves user profile data."""
        self.user_data[key] = value
        self.save_user_profile()

    def save_user_profile(self):
        """Saves the user's digital twin profile to a JSON file."""
        with open(self.user_profile_path, "w") as file:
            json.dump(self.user_data, file, indent=4)

    def learn_from_user(self, interaction):
        """Analyzes user interactions and updates the digital twin model."""
        if "behavior_patterns" not in self.user_data:
            self.user_data["behavior_patterns"] = {}
        
        if interaction in self.user_data["behavior_patterns"]:
            self.user_data["behavior_patterns"][interaction] += 1
        else:
            self.user_data["behavior_patterns"][interaction] = 1
        
        self.save_user_profile()

    def predict_user_needs(self):
        """Predicts user needs based on learned behavior patterns."""
        if not self.user_data["behavior_patterns"]:
            return "No data available to predict user needs."
        
        sorted_patterns = sorted(self.user_data["behavior_patterns"].items(), key=lambda x: x[1], reverse=True)
        most_common_action = sorted_patterns[0][0] if sorted_patterns else "unknown"
        return f"Based on your habits, you might want to {most_common_action}."

    def get_profile_summary(self):
        """Returns a summary of the digital twin's knowledge about the user."""
        return {
            "Name": self.user_data.get("name", "Unknown"),
            "Preferences": self.user_data.get("preferences", {}),
            "Top Behavior Patterns": sorted(self.user_data.get("behavior_patterns", {}).items(), key=lambda x: x[1], reverse=True)
        }

if __name__ == "__main__":
    twin = DigitalTwin()
    twin.learn_from_user("open email")
    twin.learn_from_user("open email")
    twin.learn_from_user("check news")
    print(twin.predict_user_needs())
    print(twin.get_profile_summary())
