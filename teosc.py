import sys
from src.operation_counter import OperationCounter

def main():
    if len(sys.argv) != 2:
        print("Usage: python teisc.py <python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    counter = OperationCounter()
    
    try:
        counter.count_operations(file_path)
        print(counter.get_summary())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
