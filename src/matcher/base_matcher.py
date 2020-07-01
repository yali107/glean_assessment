import os


class BaseMatcher:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

    @staticmethod
    def _check_input(pair, type_=str):
        if not all([isinstance(entity, type_)] for entity in pair):
            raise TypeError('Matcher can only match same instance type')

        return pair[0], pair[1]
