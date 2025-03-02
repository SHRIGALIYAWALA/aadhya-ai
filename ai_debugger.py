import ast
import autopep8
import traceback

class CodeDebugger:
    def __init__(self):
        print("[AI Debugger] Initialized and ready to analyze code.")
    
    def analyze_code(self, code):
        """Checks for syntax errors and suggests fixes."""
        try:
            ast.parse(code)
            print("[AI Debugger] No syntax errors detected.")
            return "No syntax errors found."
        except SyntaxError as e:
            print(f"[AI Debugger] Syntax error detected: {e}")
            return f"Syntax Error: {e}"
    
    def auto_fix_code(self, code):
        """Automatically formats and attempts to fix code issues."""
        try:
            fixed_code = autopep8.fix_code(code)
            print("[AI Debugger] Code has been auto-formatted.")
            return fixed_code
        except Exception as e:
            print(f"[AI Debugger] Error in auto-fixing: {e}")
            return f"Error: {e}"
    
    def execute_code(self, code):
        """Executes code safely and captures errors."""
        try:
            exec(code, {})  # Execute in an isolated scope
            print("[AI Debugger] Code executed successfully.")
            return "Execution Successful."
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"[AI Debugger] Error during execution: {error_trace}")
            return f"Execution Error: {error_trace}"
    
    def debug_code(self, code):
        """Full debugging process: analyze, fix, and execute."""
        print("[AI Debugger] Running full debugging process...")
        analysis = self.analyze_code(code)
        fixed_code = self.auto_fix_code(code)
        execution_result = self.execute_code(fixed_code)
        
        return {
            "analysis": analysis,
            "fixed_code": fixed_code,
            "execution_result": execution_result
        }

# Example Usage
if __name__ == "__main__":
    debugger = CodeDebugger()
    sample_code = "print('Hello, World!')"
    result = debugger.debug_code(sample_code)
    print(result)
