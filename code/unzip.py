from pathlib import Path
from typing import List, NamedTuple
from zipfile import ZipFile


class Arg(NamedTuple):
    origin: str
    destiny: str


if __name__ == '__main__':
    project_root = Path(__file__).absolute().parent.parent
    args: List[Arg] = []

    with open(project_root/'code/args/unzip.txt') as args_file:
        for line in args_file:
            args.append(Arg(*line.strip().split(' ')))

    for arg in args:
        origin = project_root/arg.origin
        destiny = project_root/arg.destiny
        print(f"Unzipping file {origin.name} to directory {destiny.name}")
        with ZipFile(origin) as zip_file:
            zip_file.extractall(destiny)
