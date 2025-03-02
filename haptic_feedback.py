import time
import random

class HapticAI:
    def __init__(self):
        self.haptic_sensors = ["Glove Sensors", "Exosuit Sensors", "Wearable Sensors"]
        self.haptic_actuators = ["Vibration Motor", "Force Feedback Motor", "Temperature Sensors"]
        print("[HapticAI] Haptic feedback system initialized.")

    def detect_touch(self):
        """Simulates detecting touch pressure from sensors."""
        sensor_data = {sensor: random.uniform(0, 1) for sensor in self.haptic_sensors}
        print(f"[HapticAI] Detected touch levels: {sensor_data}")
        return sensor_data

    def generate_feedback(self, touch_data):
        """Generates appropriate feedback based on touch intensity."""
        feedback = {}
        for sensor, intensity in touch_data.items():
            if intensity > 0.7:
                feedback[sensor] = "Strong Vibration"
            elif intensity > 0.3:
                feedback[sensor] = "Mild Vibration"
            else:
                feedback[sensor] = "No Feedback"
        print(f"[HapticAI] Generated feedback: {feedback}")
        return feedback

    def process_touch(self):
        """Main function to simulate haptic feedback processing."""
        print("[HapticAI] Processing touch feedback...")
        touch_data = self.detect_touch()
        feedback = self.generate_feedback(touch_data)
        print("[HapticAI] Haptic feedback delivered.")
        time.sleep(1)  # Simulate response delay

if __name__ == "__main__":
    haptic = HapticAI()
    for _ in range(3):
        haptic.process_touch()
