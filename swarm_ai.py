import threading
import time

class SwarmAI:
    def __init__(self, num_agents=5):
        """
        Initializes a Swarm AI system with multiple AI agents working together.
        """
        self.num_agents = num_agents
        self.agents = [f"Agent-{i+1}" for i in range(num_agents)]

    def agent_task(self, agent_name):
        """
        Simulates an agent performing a task in parallel.
        """
        print(f"{agent_name} is analyzing data...")
        time.sleep(2)
        print(f"{agent_name} completed its task!")

    def coordinate_tasks(self):
        """
        Uses multi-threading to simulate parallel execution of AI agents.
        """
        print("Swarm AI system is coordinating multiple agents...")
        threads = []
        
        for agent in self.agents:
            thread = threading.Thread(target=self.agent_task, args=(agent,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print("All agents have completed their tasks.")

if __name__ == "__main__":
    swarm_ai = SwarmAI(num_agents=5)
    swarm_ai.coordinate_tasks()
