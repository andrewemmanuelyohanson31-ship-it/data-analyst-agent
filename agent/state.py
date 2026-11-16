"""Dataset versions: the original is never modified; every cleaning step adds a version that can be undone."""
import pandas as pd


class DatasetState:
    def __init__(self, df: pd.DataFrame, name: str):
        self.name = name
        self.versions: list[tuple[str, pd.DataFrame]] = [("original", df)]

    @property
    def current(self) -> pd.DataFrame:
        return self.versions[-1][1]

    @property
    def original(self) -> pd.DataFrame:
        return self.versions[0][1]

    def apply(self, label: str, df: pd.DataFrame) -> None:
        self.versions.append((label, df))

    def undo(self) -> bool:
        if len(self.versions) == 1:
            return False
        self.versions.pop()
        return True
