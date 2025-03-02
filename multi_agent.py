import threading
import queue

class AI_Agent:
    def __init__(self, name, task_function):
        self.name = name
        self.task_function = task_function
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.running = False

    def assign_task(self, task_data):
        self.task_queue.put(task_data)
        print(f"[Multi-Agent] Task assigned to {self.name}: {task_data}")

    def run(self):
        self.running = True
        while self.running:
            try:
                task_data = self.task_queue.get(timeout=2)
                result = self.task_function(task_data)
                print(f"[Multi-Agent] {self.name} completed task: {result}")
            except queue.Empty:
                continue
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread.join()

class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, name, task_function):
        agent = AI_Agent(name, task_function)
        self.agents[name] = agent
        agent.start()
    
    def assign_task(self, agent_name, task_data):
        if agent_name in self.agents:
            self.agents[agent_name].assign_task(task_data)
        else:
            print(f"[Multi-Agent] No agent named {agent_name} found.")
    
    def stop_all(self):
        for agent in self.agents.values():
            agent.stop()

# Example agent tasks
def analyze_data(data):
    return f"Analyzed: {data}"

def generate_report(data):
    return f"Report Generated: {data}"

if __name__ == "__main__":
    multi_agent = MultiAgentSystem()
    multi_agent.register_agent("DataAnalyzer", analyze_data)
    multi_agent.register_agent("ReportGenerator", generate_report)
    
    multi_agent.assign_task("DataAnalyzer", "User Behavior Data")
    multi_agent.assign_task("ReportGenerator", "Monthly AI Performance Report")