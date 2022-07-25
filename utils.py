import os


class Trainer:
    def __init__(self):
        pass

    def train(self):
        pass


def get_name_ext(file_path: str) -> [str, str]:
    if os.sep in file_path:
        file_name = file_path.split(os.sep)[-1]
    else:
        file_name = file_path
    if os.extsep in file_name:
        name, ext = file_name.rsplit(os.extsep, maxsplit=1)
    else:
        name, ext = file_name, ""
    return name, ext
