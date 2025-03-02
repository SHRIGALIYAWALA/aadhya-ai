from features.speech import SpeechAssistant
from features.vision import VisionModule
from features.memory import MemoryManager
from features.emotion import EmotionAI
from features.predictive import PredictiveAssistant
from features.security import CyberSecurity
from features.multi_agent import MultiAgentSystem
from features.personalization import AdaptivePersonality
from features.quantum_ai import QuantumAI
from features.neural_interface import BCIController
from features.holography import HolographicInterface
from features.drone_control import DroneAI
from features.real_world_ai import RealWorldAssistant
from features.sleep_analysis import SleepMonitor
from features.simulation import SimulationEngine
from features.ethical_ai import EthicalDecisionAI
from features.ai_debugger import CodeDebugger
from features.digital_twin import DigitalTwin
from features.swarm_ai import SwarmAI
from features.haptic_feedback import HapticAI
from features.ai_prototyping_lab import AIPrototypeGUI
from features.ai_innovation_advisor import InnovationGUI
from robotic_controller import RoboticController
import threading
import tkinter as tk
import queue
import logging
import time

class ServiceOrchestrator:
    def __init__(self):
        self.services = {}
        self.event_queue = queue.Queue()
        self.logger = logging.getLogger("AadhyaAI")
        logging.basicConfig(level=logging.INFO)
        self.restart_attempts = {}
        self.max_retries = 3
        self.cooldown_period = 10

    def register_service(self, name, service):
        self.services[name] = service
        self.restart_attempts[name] = 0
        self.logger.info(f"Registered service: {name}")

    def start_service(self, name):
        try:
            service = self.services[name]
            try:
                thread = threading.Thread(target=service, daemon=True)
                thread.start()
                self.logger.info(f"Started service: {name}")
            except Exception as thread_error:
                self.logger.error(f"Thread initialization failed for service {name}: {type(thread_error).__name__} - {thread_error}")
                self.restart_service(name)
        except Exception as e:
            self.logger.error(f"Failed to start service {name}: {type(e).__name__} - {e}")
            self.restart_service(name)

    def restart_service(self, name):
        if self.restart_attempts[name] >= self.max_retries:
            self.logger.error(f"Max retries reached for service: {name}. Cooling down for {self.cooldown_period} seconds.")
            time.sleep(self.cooldown_period)
            self.restart_attempts[name] = 0
        else:
            self.restart_attempts[name] += 1
            self.logger.info(f"Restarting service: {name}, Attempt: {self.restart_attempts[name]}")
            self.start_service(name)

    def emit_event(self, event):
        self.event_queue.put(event)

    def process_events(self):
        while True:
            event = self.event_queue.get()
            try:
                self.logger.info(f"Processing event: {event}")
            finally:
                self.event_queue.task_done()

class AadhyaAI:
    def __init__(self):
        self.orchestrator = ServiceOrchestrator()
        self.modules = {
            "speech": SpeechAssistant(),
            "vision": VisionModule(),
            "memory": MemoryManager(),
            "emotion": EmotionAI(),
            "predictive": PredictiveAssistant(),
            "security": CyberSecurity(),
            "multi_agent": MultiAgentSystem(),
            "personalization": AdaptivePersonality(),
            "quantum_ai": QuantumAI(),
            "bci": BCIController(),
            "holography": HolographicInterface(),
            "drone": DroneAI(),
            "real_world_ai": RealWorldAssistant(),
            "sleep_analysis": SleepMonitor(),
            "simulation": SimulationEngine(),
            "ethical_ai": EthicalDecisionAI(),
            "debugger": CodeDebugger(),
            "digital_twin": DigitalTwin(),
            "swarm_ai": SwarmAI(),
            "haptic_feedback": HapticAI()
        }
        for name, module in self.modules.items():
            self.orchestrator.register_service(name, module.run)

    def start_ai_prototyping_lab(self):
        try:
            root = tk.Tk()
            app = AIPrototypeGUI(root)
            root.mainloop()
        except Exception as e:
            logging.error(f"AI Prototyping Lab crashed: {type(e).__name__} - {e}")

    def start_innovation_advisor(self):
        try:
            root = tk.Tk()
            app = InnovationGUI(root)
            root.mainloop()
        except Exception as e:
            logging.error(f"Innovation Advisor crashed: {type(e).__name__} - {e}")

    def run(self):
        threading.Thread(target=self.start_ai_prototyping_lab, daemon=True).start()
        threading.Thread(target=self.orchestrator.process_events, daemon=True).start()
        for name in self.modules.keys():
            self.orchestrator.start_service(name)

if __name__ == "__main__":
    aadhya = AadhyaAI()
    aadhya.run()
