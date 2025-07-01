"""
Dataset utilities – class-based rewrite of old_code.dataset and old_code.data_utils.

Public API (mirrors legacy free functions)
------------------------------------------
Dataset(data_dir, scaler=None)                   # constructor

load(date)                                       # -> x_df, y_df, master_list
filter(x_df, cols)                               # -> x_df[cols]
split(x_df, y_df, test_ids, by="cell|cycle")     # -> x_tr, y_tr, x_te, y_te
scale(x_df, fit=True)                            # -> scaled DataFrame

get_scaled_split_cell(date, test_cells, feats)   # -> x_tr, x_te, y_tr, y_te, scaler
get_scaled_split_cycle(date, test_cycles, feats) # -> x_tr, x_te, y_tr, y_te, scaler
"""

from __future__ import annotations

import os
import pickle
from typing import Iterable, Sequence, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


class Dataset:
    """Utility for loading, splitting and scaling cycle-life data."""

    # ------------------------------------------------------------------ #
    # construction / I-O
    # ------------------------------------------------------------------ #
    def __init__(self, data_dir: str, scaler_dir : str='⚖️', features: Iterable[str]=None) -> None:
        path = lambda fname: os.path.join(data_dir, fname)
        self.xCC = pd.read_pickle(path("xCC.pkl"))
        self.yCC = pd.read_pickle(path("yCC.pkl"))
        if features is not None:
            self.xCC = self.xCC[features]
        with open(path("masterList.pkl"), "rb") as fp:
            self.masterList = pickle.load(fp)
        spath = lambda fname: os.path.join(scaler_dir, fname)
        if os.path.exists(spath("scaler.pkl")):
            with open(spath("scaler.pkl"), "rb") as fp:
                self.scaler = pickle.load(fp)
        else:
            self.scaler = RobustScaler()

    def trim(self, size: Optional[Union[int, None]] = None, *, random_state: int | None = None):
        """
        Randomly subsample the dataset to the requested size.

        Parameters
        ----------
        size : int | None
            Number of rows to keep. If None, the dataset is left unchanged.
        random_state : int | None, optional
            Seed for reproducible subsampling.
        """
        if size is None:
            return

        # draw a random subset of indices and keep rows aligned across xCC and yCC
        # idx = self.xCC.sample(n=size, random_state=random_state, replace=False).index
        # self.xCC = self.xCC.loc[idx]
        # self.yCC = self.yCC.loc[idx]
        self.xCC = self.xCC.iloc[:size]
        self.yCC = self.yCC.iloc[:size]

    # ---- shifted series (legacy support) ----------------------------- #
    def add_shifted_series(
        self,
        series_data: pd.Series | None = None,
        new_col: str = None,
    ) -> pd.DataFrame:
        if series_data is None:
            series_data = self.yCC
        if new_col is None:
            base = series_data.name or "series"
            new_col = f"shifted_{base}"

        self.xCC[new_col] = np.inf

        cell_idx = self.xCC.index.get_level_values("cell_num")
        cycle_idx = self.xCC.index.get_level_values("cycle")
        uniq = set(zip(cell_idx, cycle_idx))

        s_cell = series_data.index.get_level_values("cell_num")
        s_cycle = series_data.index.get_level_values("cycle")

        for cell, cyc in tqdm(uniq, disable=True):
            mask = (cell_idx == cell) & (cycle_idx == cyc)
            pos = np.where(mask)[0]
            if len(pos) <= 1:
                continue

            s_mask = (s_cell == cell) & (s_cycle == cyc)
            if not np.any(s_mask):
                continue

            vals = series_data.iloc[np.where(s_mask)[0]].values
            shifted = np.zeros(len(pos))
            n = min(len(pos) - 1, len(vals))
            shifted[1 : n + 1] = vals[:n]
            col_loc = self.xCC.columns.get_loc(new_col)
            for i, p in enumerate(pos):
                self.xCC.iat[p, col_loc] = shifted[i]
        return self.xCC

    # ------------------------------------------------------------------ #
    # splitting
    # ------------------------------------------------------------------ #
    def add_C_rate(self):
        if 'c-rate' in self.xCC.index.names:
            return
        cell_nums = self.xCC.index.get_level_values("cell_num")
        labels = cell_nums.map(self.masterList)
        new_index = pd.MultiIndex.from_arrays(
            [
                self.xCC.index.get_level_values("cell_num"),
                self.xCC.index.get_level_values("cycle"),
                labels
            ],
            names=["cell_num", "cycle", "c-rate"]
        )
        self.xCC.index = new_index

    def split(
        self,
        test_ids: Sequence,
        by: str = "cell_num", #one of cell_num or cycle, c-rate
    ):
        if by=='c-rate':
            self.add_C_rate()
        else:
            test_ids = [int(id) for id in test_ids]
        
        cycles_all = self.xCC.index.get_level_values(by).unique()
        train = [c for c in cycles_all if c not in test_ids]

        mask_tr = self.xCC.index.get_level_values(by).isin(train)
        mask_te = self.xCC.index.get_level_values(by).isin(test_ids)

        x_tr = self.xCC.loc[mask_tr, :]
        y_tr = self.yCC.loc[mask_tr]
        x_te = self.xCC.loc[mask_te, :]
        y_te = self.yCC.loc[mask_te]

        return x_tr, y_tr, x_te, y_te

    # ------------------------------------------------------------------ #
    # scaling
    # ------------------------------------------------------------------ #
    def scale(self, x: pd.DataFrame, *, fit: bool = True) -> pd.DataFrame:
        if "shifted_NEP" in x.columns:
            x = x.copy()
            x["shifted_NEP"] += np.random.normal(0, 0.1, len(x))

        if fit:
            self.scaler.fit(x)

        x_scaled = self.scaler.transform(x)
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    # ------------------------------------------------------------------ #
    # high-level composite helpers
    # ------------------------------------------------------------------ #
    def get_scaled_split(
        self,
        test_ids: Sequence,
        by: str="cell_num",  # one of cell_num or cycle
    ):
        x_tr, y_tr, x_te, y_te = self.split(test_ids, by)
        x_tr = self.scale(x_tr, fit=True)
        x_te = self.scale(x_te, fit=False)
        return x_tr.values, x_te.values, np.array(y_tr), np.array(y_te), self.scaler
