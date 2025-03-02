from .speech import SpeechAssistant
from .vision import VisionModule
from .memory import MemoryManager
from .emotion import EmotionAI
from .predictive import PredictiveAssistant
from .security import CyberSecurity
from .multi_agent import MultiAgentSystem
from .personalization import AdaptivePersonality
from .quantum_ai import QuantumAI
from .neural_interface import BCIController
from .holography import HolographicInterface
from .drone_control import DroneAI
from .real_world_ai import RealWorldAssistant
from .sleep_analysis import SleepMonitor
from .simulation import SimulationEngine
from .ethical_ai import EthicalDecisionAI
from .ai_debugger import CodeDebugger
from .digital_twin import DigitalTwin
from .swarm_ai import SwarmAI
from .haptic_feedback import HapticAI

# Dynamically initialize all features in a dictionary
def initialize_features():
    return {
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
        "haptic_feedback": HapticAI(),
    }
