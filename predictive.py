import datetime
import random

class PredictiveAssistant:
    def __init__(self):
        self.user_history = []  # Stores past user interactions
        self.daily_tasks = {}

    def log_interaction(self, action):
        """Logs user interactions for pattern analysis."""
        timestamp = datetime.datetime.now()
        self.user_history.append((timestamp, action))

    def anticipate_user_needs(self):
        """Predicts the user's next action based on historical data."""
        if not self.user_history:
            print("No user history available for prediction.")
            return
        
        actions = [entry[1] for entry in self.user_history]
        predicted_action = max(set(actions), key=actions.count)  # Most frequent action
        print(f"Predictive AI: Based on past behavior, you might want to {predicted_action} now.")
    
    def schedule_task(self, task, time):
        """Schedules a task at a specific time."""
        self.daily_tasks[time] = task
        print(f"Task '{task}' scheduled for {time}.")
    
    def check_scheduled_tasks(self):
        """Checks and notifies about upcoming tasks."""
        now = datetime.datetime.now().strftime('%H:%M')
        if now in self.daily_tasks:
            print(f"Reminder: It's time for {self.daily_tasks[now]}!")
        
    def recommend_activity(self):
        """Suggests activities based on time and context."""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 9:
            print("Good morning! How about some light stretching or a morning walk?")
        elif 9 <= hour < 12:
            print("Time to focus! Maybe start your work or study session.")
        elif 12 <= hour < 14:
            print("Lunch time! A healthy meal will boost your energy.")
        elif 14 <= hour < 18:
            print("Afternoon productivity boost! Try completing pending tasks.")
        elif 18 <= hour < 22:
            print("Evening relaxation time! Maybe read a book or listen to music.")
        else:
            print("It's late! Consider getting some rest for a productive tomorrow.")

if __name__ == "__main__":
    predictive = PredictiveAssistant()
    predictive.log_interaction("check emails")
    predictive.anticipate_user_needs()
    predictive.schedule_task("Workout", "18:30")
    predictive.check_scheduled_tasks()
    predictive.recommend_activity()
