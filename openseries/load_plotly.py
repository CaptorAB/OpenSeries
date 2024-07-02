"""Function to load plotly layout and configuration from local json file."""

from __future__ import annotations

from json import load
from logging import warning
from pathlib import Path

import requests
from requests.exceptions import ConnectionError

from openseries.types import CaptorLogoType, PlotlyLayoutType

__all__ = ["load_plotly_dict"]


def _check_remote_file_existence(url: str) -> bool:
    """
    Check if remote file exists.

    Parameters
    ----------
    url: str
        Path to remote file

    Returns
    -------
    bool
        True if url is valid and False otherwise

    """
    ok_code = 200

    try:
        response = requests.head(url, timeout=30)
        if response.status_code != ok_code:
            return False
    except ConnectionError:
        return False
    return True


def load_plotly_dict(
    *,
    responsive: bool = True,
) -> tuple[PlotlyLayoutType, CaptorLogoType]:
    """
    Load Plotly defaults.

    Parameters
    ----------
    responsive : bool
        Flag whether to load as responsive

    Returns
    -------
    tuple[PlotlyLayoutType, CaptorLogoType]
        A dictionary with the Plotly config and layout template

    """
    project_root = Path(__file__).resolve().parent.parent
    layoutfile = project_root.joinpath("openseries").joinpath("plotly_layouts.json")
    logofile = project_root.joinpath("openseries").joinpath("plotly_captor_logo.json")

    with Path.open(layoutfile, mode="r", encoding="utf-8") as layout_file:
        fig = load(layout_file)
    with Path.open(logofile, mode="r", encoding="utf-8") as logo_file:
        logo = load(logo_file)

    if _check_remote_file_existence(url=logo["source"]) is False:
        msg = f"Failed to add logo image from URL {logo['source']}"
        warning(msg)
        logo = {}

    fig["config"].update({"responsive": responsive})

    return fig, logo
