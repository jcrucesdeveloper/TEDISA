import sys
from src.operation_counter import OperationCounter

def main():
    if len(sys.argv) != 2:
        print("[ERROR] Usage: python teisc.py <python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    counter = OperationCounter()
    print('[INFO] Loading Pytorch Tensor Operations...')
    counter.load_operations('data/pytorch_operations.txt')
    
    try:
        counter.count_operations(file_path)
        print('\n[RESULTS] Operation Summary:')
        print('-' * 50)
        print(counter.get_summary())
        print('-' * 50)
    except FileNotFoundError:
        print(f'[ERROR] File not found: {file_path}')
        sys.exit(1)
    except Exception as e:
        print(f'[ERROR] Analysis failed: {str(e)}')
        sys.exit(1)

if __name__ == "__main__":
    main()
