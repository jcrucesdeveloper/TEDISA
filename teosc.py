import sys
import os
import argparse
from src.operation_counter import OperationCounter

def print_summary(counter: OperationCounter, scanned_files: list, output_file: str = None) -> None:
    """Print the total operation summary to stdout and optionally to a file"""
    summary = f'[RESULTS] Total Operation Summary:\n'
    summary += '-' * 50 + '\n'
    summary += f'Files scanned ({len(scanned_files)}):\n'
    for file in scanned_files:
        summary += f'  - {file}\n'
    summary += '-' * 50 + '\n'
    summary += counter.get_summary()
    
    print(summary)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(summary)
            print(f'[INFO] Results written to: {output_file}')
        except Exception as e:
            print(f'[ERROR] Failed to write results to file: {str(e)}')

def process_file(counter: OperationCounter, file_path: str) -> None:
    """Process a single Python file"""
    try:
        print(f'[INFO] Counting Tensor Operations in File: {file_path}')
        counter.count_operations(file_path)
        return True
    except Exception as e:
        print(f'[ERROR] Analysis failed for {file_path}: {str(e)}')
        return False

def process_directory(counter: OperationCounter, dir_path: str) -> list:
    """Process all Python files in a directory recursively"""
    scanned_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if process_file(counter, file_path):
                    scanned_files.append(file_path)
    return scanned_files

def main():
    parser = argparse.ArgumentParser(description='Count PyTorch tensor operations in Python files')
    parser.add_argument('path', help='Path to Python file or directory to analyze')
    parser.add_argument('-o', '--output', help='Path to output file for results')
    args = parser.parse_args()
    
    counter = OperationCounter()
    print('[INFO] Loading Pytorch Tensor Operations...')
    counter.load_operations('data/pytorch_operations.txt')
    
    try:
        scanned_files = []
        if os.path.isfile(args.path):
            if process_file(counter, args.path):
                scanned_files.append(args.path)
        elif os.path.isdir(args.path):
            scanned_files = process_directory(counter, args.path)
        else:
            print(f'[ERROR] Path not found: {args.path}')
            sys.exit(1)
            
        print_summary(counter, scanned_files, args.output)
    except Exception as e:
        print(f'[ERROR] Analysis failed: {str(e)}')
        sys.exit(1)

if __name__ == "__main__":
    main()
