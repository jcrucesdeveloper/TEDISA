import sys
import os
from src.operation_counter import OperationCounter

def print_summary(counter: OperationCounter) -> None:
    """Print the total operation summary"""
    print(f'\n[RESULTS] Total Operation Summary:')
    print('-' * 50)
    print(counter.get_summary())
    print('-' * 50)

def process_file(counter: OperationCounter, file_path: str) -> None:
    """Process a single Python file"""
    try:
        print(f'[INFO] Counting Tensor Operations in File: {file_path}')
        counter.count_operations(file_path)
    except Exception as e:
        print(f'[ERROR] Analysis failed for {file_path}: {str(e)}')

def process_directory(counter: OperationCounter, dir_path: str) -> None:
    """Process all Python files in a directory recursively"""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                process_file(counter, file_path)

def main():
    if len(sys.argv) != 2:
        print("[ERROR] Usage: python teisc.py <python_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    counter = OperationCounter()
    print('[INFO] Loading Pytorch Tensor Operations...')
    counter.load_operations('data/pytorch_operations.txt')
    
    try:
        if os.path.isfile(path):
            process_file(counter, path)
        elif os.path.isdir(path):
            process_directory(counter, path)
        else:
            print(f'[ERROR] Path not found: {path}')
            sys.exit(1)
            
        print_summary(counter)
    except Exception as e:
        print(f'[ERROR] Analysis failed: {str(e)}')
        sys.exit(1)

if __name__ == "__main__":
    main()
