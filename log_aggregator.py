# ypo@informatik.uni-kiel.de

from io import StringIO
from typing import Callable, Iterable, Sequence, TypeVar
import pandas as pd

T = TypeVar('T')  # Can be anything

def stripped_lines(filename) -> Iterable[str]:
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

def filter_lines(substring: str, lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        if substring in line:
            yield line

def strip_prefix_from_lines(
        prefix: str,
        lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        yield line.removeprefix(prefix)

def split_lines(
        separator: str,
        lines: Iterable[str]) -> Iterable[Sequence[str]]:
    for line in lines:
        yield line.split(sep = separator)

def process_parts(
        processing_fun: Callable[[str], str],
        parts_iter: Iterable[Sequence[str]]
    ) -> Iterable[Sequence[str]]:
    for parts in parts_iter:
        yield [processing_fun(part) for part in parts]

def join_parts(
        parts_iter: Iterable[Sequence[str]],
        sep = ',') -> Iterable[str]:
    for parts in parts_iter:
        yield sep.join(parts)

def to_dataframe(
        lines: Iterable[str],
        names,
        index_col = None,
        sep = ',') -> pd.DataFrame:
    lines = StringIO('\n'.join(lines))
    df = pd.read_csv(
        lines,
        sep = sep,
        header = None,
        names = names,
        index_col = index_col
    )
    return df

def loss_evolution_extractor(filename):
    lines = stripped_lines(filename)
    lines = filter_lines("#loss_evolution#", lines)
    lines = strip_prefix_from_lines(
        "INFO:model:#loss_evolution# --- fitting decision ",
        lines
    )
    parts_iter = split_lines(';', lines)
    parts_iter = process_parts(loss_evolution_worker, parts_iter)
    lines = join_parts(parts_iter)
    df = to_dataframe(
        lines,
        names = ['tree_idx', 'prev_loss', 'current_loss'],
        index_col = 'tree_idx'
    )
    return df

def loss_evolution_worker(part: str) -> str:
    part = part.lstrip()
    if part.startswith("tree "):
        return part.removeprefix("tree ")
    elif part.startswith("previous loss: "):
        return part.removeprefix("previous loss: ")
    elif part.startswith("current loss: "):
        return part.removeprefix("current loss: ")
    else:
        raise ValueError(
            "String part was not as expected. Part: {}".format(part)
        )

def tree_evolution_extractor(filename):
    lines = stripped_lines(filename)
    lines = filter_lines("#tree_evolution#", lines)
    lines = strip_prefix_from_lines(
        "INFO:model:#tree_evolution# --- ensemble ",
        lines
    )
    parts_iter = split_lines(" decision tree ", lines)
    parts_iter = process_parts(tree_evolution_worker, parts_iter)
    lines = join_parts(parts_iter)
    df = to_dataframe(
        lines,
        names = ['included?', 'tree_idx'],
        index_col = 'tree_idx'
    )
    return df

def tree_evolution_worker(part: str) -> str:
    if part == 'includes':
        return "True"
    elif part == 'excludes':
        return "False"
    else:
        return part

if __name__ == '__main__':
    df = loss_evolution_extractor("in_depth_analysis_1.log")
    print(df)
    df = tree_evolution_extractor("in_depth_analysis_1.log")
    print(df)
