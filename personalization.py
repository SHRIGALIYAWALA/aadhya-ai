import random

class AdaptivePersonality:
    def __init__(self):
        self.mood = "neutral"
        self.speech_tone = "normal"
        self.user_preferences = {}

    def analyze_user_interaction(self, user_input):
        """Analyzes user input to detect sentiment and adjust personality."""
        positive_keywords = ["happy", "excited", "love", "great", "awesome"]
        negative_keywords = ["sad", "angry", "upset", "bad", "frustrated"]
        
        if any(word in user_input.lower() for word in positive_keywords):
            self.mood = "positive"
            self.speech_tone = "cheerful"
        elif any(word in user_input.lower() for word in negative_keywords):
            self.mood = "negative"
            self.speech_tone = "calm and soothing"
        else:
            self.mood = "neutral"
            self.speech_tone = "normal"

    def adjust_behavior(self):
        """Adjusts AI's response style based on detected mood."""
        responses = {
            "positive": ["I'm glad to hear that! Let's keep the good vibes going!", "Awesome! How can I help you today?"],
            "negative": ["I understand, I'm here to support you.", "Take a deep breath. What can I do to help?"],
            "neutral": ["Got it. What’s next?", "Tell me more about that."]
        }
        
        return random.choice(responses[self.mood])

    def update_preferences(self, key, value):
        """Stores user preferences for a personalized experience."""
        self.user_preferences[key] = value

    def get_personalized_response(self, user_input):
        """Processes input, updates AI's mood, and returns a personalized response."""
        self.analyze_user_interaction(user_input)
        return self.adjust_behavior()

# Example Usage
if __name__ == "__main__":
    personality = AdaptivePersonality()
    user_input = input("How are you feeling today? ")
    response = personality.get_personalized_response(user_input)
    print(response)
