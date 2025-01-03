import os
import argparse
from pathlib import Path

def rename_files(directory: str, base_name: str):
    # Convert to Path and validate directory
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist or is not a directory")
    
    # Get all files and sort them
    files = sorted([f for f in dir_path.iterdir() if f.is_file()])
    
    # Incase script is within the directory, remove it from the list
    files = [f for f in files if f.name != 'rename_files.py']
    
    # Rename files with new pattern
    for index, file in enumerate(files, 1):
        extension = file.suffix
        new_name = f'{base_name}{index:02d}{extension}'
        file.rename(Path(dir_path, new_name))
        print(f'Renamed: {file.name} -> {new_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename files in a directory with sequential numbering')
    parser.add_argument('--directory', '-d', required=True, help='Directory containing files to rename')
    parser.add_argument('--base-name', '-b', default='IMG_', help='Base name for renamed files (default: IMG_)')
    
    args = parser.parse_args()
    rename_files(args.directory, args.base_name)