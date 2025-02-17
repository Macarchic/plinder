# Copyright (c) 2024, Plinder Development Team
# Distributed under the terms of the Apache License 2.0
from __future__ import annotations

import pandas as pd
from duckdb import sql

from plinder_fork.core.scores.query import FILTERS, make_query
from plinder_fork.core.utils import cpl
from plinder_fork.core.utils.config import get_config
from plinder_fork.core.utils.log import setup_logger

import functools


LOG = setup_logger(__name__)

def ensure_config_loaded():
    """
     Decorator that ensures the configuration (`cfg`) is loaded before the function execution.
    If `cfg` is provided as an argument, it is used; otherwise, a default configuration is loaded.

    The configuration (`cfg`) should be passed as the `cfg` argument in `query_index`.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            internal_args = {'cfg'}
            # Retrieve cfg from arguments if provided, otherwise load default config
            cfg_file = kwargs.get("cfg", None)
            cfg = get_config(config_file=cfg_file)
            LOG.info(f"config: {cfg.data.plinder_iteration}")
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in internal_args}

            try:
                return func(*args, **filtered_kwargs)
            except Exception as e:
                LOG.error(f"Сталася непередбачена помилка: {e}", exc_info=True)
                raise  
        return wrapper
    return decorator

@ensure_config_loaded
def query_index(
    *,
    columns: list[str] | None = None,
    splits: list[str] | None = None,
    filters: FILTERS = None,
) -> pd.DataFrame:
    """
    Query the index database.

    Parameters
    ----------
    columns : list[str], default=None
        the columns to return
    filters : list[tuple[str, str, str]]
        the filters to apply

    Returns
    -------
    df : pd.DataFrame | None
        the index results
    """
    cfg = get_config()
    dataset = cpl.get_plinder_path(rel=f"{cfg.data.index}/{cfg.data.index_file}")
    if columns is None:
        columns = ["system_id", "entry_pdb_id"]
    if "system_id" not in columns and "*" not in columns:
        columns = ["system_id"] + columns
    # TODO: remove this patch after binding_affinity is fixed
    # START patch
    if "system_has_binding_affinity" in columns or "ligand_binding_affinity" in columns:
        raise ValueError(
            "columns containing binding_affinity have been removed until bugfix"
            "see: https://github.com/plinder-org/plinder/issues/94"
        )
    # END patch
    query = make_query(
        dataset=dataset,
        columns=columns,
        filters=filters,
        allow_no_filters=True,
    )
    assert query is not None
    df = sql(query).to_df()
    if splits is None:
        splits = ["train", "val"]
    split = cpl.get_plinder_path(rel=f"{cfg.data.splits}/{cfg.data.split_file}")
    split_df = pd.read_parquet(split)
    split_dict = dict(zip(split_df["system_id"], split_df["split"]))
    df["split"] = df["system_id"].map(lambda x: split_dict.get(x, "unassigned"))
    if "*" not in splits:
        df = df[df["split"].isin(splits)].reset_index(drop=True)
    return df
