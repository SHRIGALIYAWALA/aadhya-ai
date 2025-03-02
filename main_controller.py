import importlib
import threading
import queue
import logging
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from flask import Flask, jsonify, request
from robotic_controller import RoboticController


app = Flask(__name__)

class ServiceOrchestrator:
    def __init__(self):
        self.services = {}
        self.event_queue = queue.Queue()
        self.logger = logging.getLogger("AadhyaAI")
        logging.basicConfig(level=logging.INFO)
        self.restart_attempts = defaultdict(int)
        self.max_retries = 3
        self.cooldown_period = 10
        self.health_checks = defaultdict(int)
        self.service_logs = defaultdict(list)

    def register_service(self, name, service):
        self.services[name] = service
        self.restart_attempts[name] = 0
        self.health_checks[name] = 0
        self.logger.info(f"Registered service: {name}")

    def start_service(self, name):
        try:
            service = self.services[name]
            try:
                thread = threading.Thread(target=service, daemon=True)
                thread.start()
                self.logger.info(f"Started service: {name}")
                self.monitor_health(name)
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
                self.service_logs[event["service"]].append(event["status"])
            finally:
                self.event_queue.task_done()

    def monitor_health(self, name):
        def health_check():
            while True:
                time.sleep(30)
                self.health_checks[name] += 1
                if self.health_checks[name] > 3:
                    self.logger.warning(f"Health check failed for {name}. Restarting service.")
                    self.restart_service(name)
                    self.health_checks[name] = 0

        threading.Thread(target=health_check, daemon=True).start()

    def visualize_logs(self):
        for service, logs in self.service_logs.items():
            plt.plot(range(len(logs)), logs, label=service)
        plt.xlabel("Events")
        plt.ylabel("Status")
        plt.title("Service Performance Logs")
        plt.legend()
        plt.draw()  # Non-blocking plot rendering
        plt.pause(0.001)  # Small pause to allow GUI events
        print("Visualization rendered without blocking.")


    def get_service_status(self):
        status = {name: ("Running" if self.health_checks[name] <= 3 else "Failed") for name in self.services.keys()}
        return status

class AadhyaMasterController:
    def __init__(self):
        self.orchestrator = ServiceOrchestrator()
        self.robotic_controller = RoboticController()  # Add this line
        self.module_names = [
            'drone_control', 'ai_prototyping_lab', 'ai_innovation_advisor', 
            'emotion', 'ai_debugger', 'digital_twin', 'ethical_ai', 
            'holography', 'haptic_feedback', 'swarm_ai', 'memory', 
            'multi_agent', 'neural_interface', 'personalization', 
            'predictive', 'quantum_ai', 'real_world_ai', 'security', 
            'simulation', 'speech', 'vision'
        ]
        self.initialize_modules()
        threading.Thread(target=self.orchestrator.process_events, daemon=True).start()


    def initialize_modules(self):
        self.orchestrator.register_service("robotic_controller", self.robotic_controller.init_mqtt)  # Add this
        for module in self.module_names:
            try:
                imported_module = importlib.import_module(f"features.{module}")
                class_name = ''.join([part.capitalize() for part in module.split('_')])
                instance = getattr(imported_module, class_name)()
                self.orchestrator.register_service(module, instance.run)
                print(f"{module} initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize {module}: {e}")


    def start_all_services(self):
        for module in self.module_names:
            self.orchestrator.start_service(module)
            threading.Thread(target=self.robotic_controller.health_check, daemon=True).start()


@app.route("/status", methods=["GET"])
def status():
    return jsonify(aadhya.orchestrator.get_service_status())

@app.route("/restart/<service>", methods=["POST"])
def restart(service):
    if service in aadhya.module_names:
        aadhya.orchestrator.restart_service(service)
        return jsonify({"message": f"Service {service} restarted successfully"})
    return jsonify({"error": "Service not found"}), 404

if __name__ == "__main__":
    aadhya = AadhyaMasterController()
    aadhya.start_all_services()
    threading.Thread(target=lambda: app.run(debug=True, port=5000, use_reloader=False), daemon=True).start()
    time.sleep(60)  # Allow some services to log events
    aadhya.orchestrator.visualize_logs()
