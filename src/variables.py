from src.utils import load_yaml


class AppWideVariables:
    def __init__(self):
        self.variables = load_yaml("variables.yaml")
