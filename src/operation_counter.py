import ast
from typing import Dict

class OperationCounter:
    def __init__(self):
        self.operations = {}
    
    def load_operations(self, file_path: str) -> Dict[str, int]:
        """
        Load PyTorch operations from a file
        """
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Store both the full name and the function name without nn.
                    self.operations[line] = 0
                    if line.startswith('nn.'):
                        self.operations[line[3:]] = 0

    def count_operations_from_string(self, content: str) -> Dict[str, int]:
        """
        Count operations from a string containing Python code
        """
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Handle nn.operations
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'nn':
                        func_name = f"nn.{node.func.attr}"
                        if func_name in self.operations:
                            self.operations[func_name] += 1
                    # Handle direct function calls
                    func_name = node.func.attr.lower()
                    if func_name in self.operations:
                        self.operations[func_name] += 1
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    if func_name in self.operations:
                        self.operations[func_name] += 1
        
        return self.operations

    def count_operations(self, file_path: str) -> Dict[str, int]:
        """
        Count operations in a Python file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.count_operations_from_string(content)
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of the operations sorted by count in descending order
        """
        summary = "Operation Count Summary:\n"
        sorted_ops = sorted(self.operations.items(), key=lambda x: x[1], reverse=True)
        for op, count in sorted_ops:
            summary += f"{op}: {count}\n"
        return summary 