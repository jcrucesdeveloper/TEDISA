import ast
from typing import Dict

class OperationCounter:
    def __init__(self):
        self.operations = {}
    
    def load_operations(self, file_path: str) -> Dict[str, int]:
        """
        Load PyTorch operations from a file
        """
    

    def count_operations(self, file_path: str) -> Dict[str, int]:
        """
        Count operations in a Python file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr.lower()
                    if func_name in self.operations:
                        self.operations[func_name] += 1
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    if func_name in self.operations:
                        self.operations[func_name] += 1
        
        return self.operations
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of the operations
        """
        summary = "Operation Count Summary:\n"
        for op, count in self.operations.items():
            summary += f"{op}: {count}\n"
        return summary 