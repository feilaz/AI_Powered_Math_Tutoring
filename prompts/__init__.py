import os

def load_prompt(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r') as file:
        return file.read().strip()