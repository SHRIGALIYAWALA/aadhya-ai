import tkinter as tk
from tkinter import scrolledtext
import openai
import threading

# Set your OpenAI API Key
OPENAI_API_KEY = "your-api-key-here"
openai.api_key = OPENAI_API_KEY

class AIInnovationAdvisor:
    def __init__(self):
        pass

    def generate_innovation(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

class InnovationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Driven Innovation Advisor 🚀")
        self.root.geometry("800x600")
        
        self.ai = AIInnovationAdvisor()

        self.label = tk.Label(root, text="Select an Innovation Task:", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)

        self.options = [
            "Next-Gen Device Predictor",
            "AI-Powered Problem Solver",
            "Sustainable Tech Generator",
            "Alternative Energy Innovator"
        ]

        self.selection = tk.StringVar(root)
        self.selection.set(self.options[0])  # Default selection
        self.dropdown = tk.OptionMenu(root, self.selection, *self.options)
        self.dropdown.pack(pady=5)

        self.button = tk.Button(root, text="Generate Innovation", command=self.get_innovation, font=("Arial", 12, "bold"))
        self.button.pack(pady=10)

        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Arial", 12))
        self.output_box.pack(pady=10)

    def get_innovation(self):
        task = self.selection.get()
        prompts = {
            "Next-Gen Device Predictor": "Predict the next big technological breakthroughs in AI, robotics, and computing.",
            "AI-Powered Problem Solver": "Suggest new inventions to solve global challenges like climate change, energy crisis, and medical issues.",
            "Sustainable Tech Generator": "Generate ideas for eco-friendly, sustainable devices that reduce waste and use renewable energy.",
            "Alternative Energy Innovator": "Develop new energy-harvesting methods like quantum batteries, kinetic energy, and space-based solar power."
        }
        
        prompt = prompts[task]

        def generate_response():
            result = self.ai.generate_innovation(prompt)
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, result)

        threading.Thread(target=generate_response, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = InnovationGUI(root)
    root.mainloop()
