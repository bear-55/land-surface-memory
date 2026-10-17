"""CAMELS forcing, protected development labels, and causal windows."""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DYNAMIC_COLUMNS = ("prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)")


def read_forcing(path: Path):
    if not path.name.endswith("_lump_cida_forcing_leap.txt"):
        raise ValueError("Expected original Daymet forcing filename")
    with path.open(encoding="utf-8-sig") as stream:
        float(stream.readline())
        float(stream.readline())
        area_m2 = float(stream.readline())
        header = stream.readline().split()
        required = ("Year", "Mnth", "Day", *DYNAMIC_COLUMNS)
        if len(set(header)) != len(header) or not set(required).issubset(header):
            raise ValueError("Forcing fields missing or duplicated")
        frame = pd.read_csv(stream, sep=r"\s+", names=header, usecols=list(required))
    dates = pd.to_datetime(frame[["Year", "Mnth", "Day"]].rename(columns={"Year": "year", "Mnth": "month", "Day": "day"})).to_numpy().astype("datetime64[D]")
    if len(dates) == 0 or np.any(np.diff(dates).astype(int) != 1):
        raise ValueError("Forcing dates must be unique, ordered and daily")
    values = frame[list(DYNAMIC_COLUMNS)].to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    valid &= (values[:, 0] >= 0) & (values[:, 1] >= 0) & (values[:, 4] >= 0)
    valid &= (values[:, 3] > -999) & (values[:, 2] >= values[:, 3])
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("Invalid forcing-header area")
    return dates, values, valid, area_m2


class DevelopmentStreamflow:
    """Read train/validation labels while rejecting held-out test basins."""
    def __init__(self, files, splits, periods):
        all_ids = sum([list(splits[r]) for r in ("train", "val", "test")], [])
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Overlapping basin splits")
        self.files = files
        self.splits = {key: set(value) for key, value in splits.items()}
        self.periods = periods
        self.accesses = []

    def read(self, basin_id: str, role: str, area_m2: float):
        if role not in ("train", "val") or basin_id in self.splits["test"]:
            raise PermissionError("Final-test streamflow is sealed during development")
        if basin_id not in self.splits[role]:
            raise PermissionError("Basin is not in the requested frozen role")
        if not np.isfinite(area_m2) or area_m2 <= 0:
            raise ValueError("Runoff conversion needs a finite positive fixed area")
        start, end = [date.fromisoformat(str(day)) for day in self.periods[role]]
        dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"))
        values = np.full(len(dates), np.nan, dtype=np.float64)
        seen, flags = set(), Counter()
        previous = None
        path = Path(self.files[basin_id])
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 6 or fields[0] != basin_id:
                    raise ValueError(f"Unexpected Q schema at {basin_id}:{line_number}")
                day = date(*map(int, fields[1:4]))
                if previous is not None and day <= previous:
                    raise ValueError("Q dates are duplicated or unordered")
                previous = day
                if day < start:
                    continue
                if day > end:
                    break  # Do not parse future-period Q values.
                seen.add(day)
                flag = fields[5]
                flags[flag] += 1
                # Official six-column files contain A, qualifiers such as A:e, or M.
                if not (flag == "M" or flag.startswith("A") or flag.startswith("P")):
                    raise ValueError(f"Unreviewed streamflow flag: {flag}")
                observed = float(fields[4])
                if flag == "M" or not np.isfinite(observed) or observed < 0:
                    continue
                values[(day-start).days] = observed * 0.028316846592 * 86400 * 1000 / area_m2
        self.accesses.append({"basin_id": basin_id, "role": role, "period": [str(start), str(end)],
                              "days_present": len(seen), "valid_q_days": int(np.isfinite(values).sum()),
                              "flags": dict(flags), "area_m2": area_m2})
        return dates, values


def valid_window_ends(valid_days, dates, start, end, length, q=None):
    """Every daily input in [t-L+1,t] must be valid and within the same basin."""
    valid_days = np.asarray(valid_days, dtype=bool)
    if len(valid_days) != len(dates) or length < 1:
        raise ValueError("Invalid window arguments")
    if len(dates) > 1 and np.any(np.diff(dates).astype(int) != 1):
        raise ValueError("Window index requires a continuous daily calendar")
    prefix = np.r_[0, np.cumsum(valid_days, dtype=np.int64)]
    ends = np.arange(length - 1, len(valid_days))
    complete = prefix[ends + 1] - prefix[ends + 1 - length] == length
    allowed = (dates[ends] >= np.datetime64(str(start))) & (dates[ends] <= np.datetime64(str(end)))
    if q is not None:
        allowed &= np.isfinite(q[ends]) & (q[ends] >= 0)
    return ends[complete & allowed]


class CamelsWindows(Dataset):
    """Load only cached source/validation supervision; test forcing is inference-only."""
    def __init__(self, cache_path, metadata, role="train"):
        if role not in ("train", "val"):
            raise PermissionError("No test-label dataset in the development pipeline")
        self.role = role
        with np.load(cache_path, allow_pickle=False) as saved:
            self.dynamic = saved["x_dynamic"]
            self.static = saved["x_static"]
            self.valid = saved["dynamic_valid"]
            self.dates = saved["dates"]
            self.all_ids = saved["basin_ids"].tolist()
            self.q = saved["q_" + role]
        self.ids = metadata["splits"][role]
        self.basin_rows = [self.all_ids.index(basin_id) for basin_id in self.ids]
        self.length = metadata["sequence_length"]
        self.land_indices = metadata["land_indices"]
        self.sigma = metadata["sigma_train_mm"] if role == "train" else None
        start, end = metadata["label_periods"][role]
        self.ends = [valid_window_ends(self.valid[row], self.dates, start, end, self.length, self.q[j])
                     for j, row in enumerate(self.basin_rows)]
        if any(len(indices) == 0 for indices in self.ends):
            raise ValueError("A basin has no valid supervised windows; never silently discard it")
        self.cumulative = np.cumsum([len(indices) for indices in self.ends])

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        j = int(np.searchsorted(self.cumulative, index, side="right"))
        position = index - (int(self.cumulative[j-1]) if j else 0)
        return self.at(j, int(self.ends[j][position]))

    def at(self, j, t):
        row = self.basin_rows[j]
        dynamic = self.dynamic[row, t-self.length+1:t+1]
        if dynamic.shape[0] != self.length or not np.isfinite(dynamic).all() or not np.isfinite(self.q[j, t]):
            raise ValueError("Invalid window requested")
        static = self.static[row]
        result = {"x_dynamic": torch.from_numpy(dynamic.copy()), "x_static": torch.from_numpy(static.copy()),
                  "x_land": torch.from_numpy(static[self.land_indices].copy()),
                  "q_obs_mm": torch.tensor([self.q[j, t]], dtype=torch.float32),
                  "basin_id": self.ids[j], "target_date": str(self.dates[t])}
        if self.role == "train":
            result["sigma_train_mm"] = torch.tensor([self.sigma[self.ids[j]]], dtype=torch.float32)
        return result

    def paired_epoch_indices(self, seed, epoch, count=None):
        """Uniform basins, then uniform valid windows; independent of model RNG."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 617]))
        count = len(self) if count is None else count
        basins = rng.integers(0, len(self.ids), size=count)
        offsets = np.r_[0, self.cumulative[:-1]]
        return np.array([offsets[j] + rng.integers(len(self.ends[j])) for j in basins], dtype=np.int64)
