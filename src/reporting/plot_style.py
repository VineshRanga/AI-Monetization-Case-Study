"""Centralized plot style helper for director-level quant quality."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

# Consistent color palette
COLORS = {
    'steelblue': '#4682B4',
    'darkorange': '#FF8C00',
    'red': '#DC143C',
    'green': '#228B22',
    'gray': '#808080',
    'lightgrey': '#D3D3D3',
}


def apply_style():
    """Set matplotlib rcParams for consistent, readable base style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.dpi': 300,
    })


def add_title_subtitle(fig: Figure, title: str, subtitle: Optional[str] = None) -> None:
    """
    Add title and subtitle to figure using ONLY fig.suptitle (no ax.set_title).
    Ensures no overlap with axes.
    
    Args:
        fig: Matplotlib figure
        title: Main title (fontsize 13, bold)
        subtitle: Subtitle text (fontsize 10, italic, 1 line max)
    """
    # Main title via suptitle (reserve space)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Subtitle via fig.text (if provided, 1 line max)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', va='top', 
                fontsize=10, style='italic', alpha=0.8, transform=fig.transFigure)
    
    # Reserve space at top (adjust based on subtitle presence)
    # Get current subplots_adjust values or use defaults
    current_top = getattr(fig, '_current_top', 0.90)
    top_margin = 0.88 if subtitle else 0.92
    fig._current_top = max(current_top, top_margin)


def add_source(fig: Figure, source_text: str) -> None:
    """
    Add source footnote OUTSIDE axes area (never clips).
    
    Args:
        fig: Matplotlib figure
        source_text: Source text (fontsize 8, italic)
    """
    fig.text(0.01, 0.02, source_text, ha='left', va='bottom',
            fontsize=8, style='italic', alpha=0.7, transform=fig.transFigure)
    
    # Reserve space at bottom
    current_bottom = getattr(fig, '_current_bottom', 0.11)
    fig._current_bottom = max(current_bottom, 0.12)


def save_fig(fig: Figure, path: Path) -> None:
    """
    Save figure with consistent settings and close.
    Guarantees no clipping with proper padding.
    
    Args:
        fig: Matplotlib figure
        path: Output path
    """
    # Ensure white background
    fig.patch.set_facecolor('white')
    
    # Apply final spacing adjustments if set
    top = getattr(fig, '_current_top', 0.90)
    bottom = getattr(fig, '_current_bottom', 0.12)
    
    # Get current subplots_adjust values
    current_adjust = {}
    if hasattr(fig, 'subplots_adjust_kwargs'):
        current_adjust = fig.subplots_adjust_kwargs
    else:
        # Try to get from axes
        if fig.axes:
            # Use defaults
            current_adjust = {'left': 0.10, 'right': 0.95, 'top': top, 'bottom': bottom}
    
    # Apply spacing
    fig.subplots_adjust(
        top=top,
        bottom=bottom,
        left=current_adjust.get('left', 0.10),
        right=current_adjust.get('right', 0.95)
    )
    
    # Save with high DPI and tight bbox
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, 
        dpi=200, 
        bbox_inches='tight', 
        pad_inches=0.25,
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)


def format_date_axis(ax: Axes, min_tick: str = "quarterly", freq: str = "MS", rotation: int = 30) -> None:
    """
    Format date axis with readable ticks (quarterly/monthly, not weekly).
    Prevents tick label collisions.
    
    Args:
        ax: Matplotlib axes
        min_tick: Minimum tick interval ("quarterly" or "monthly")
        freq: Frequency string (default "MS" = month start)
        rotation: Label rotation in degrees (default 30)
    """
    if min_tick == "quarterly":
        locator = mdates.MonthLocator(interval=3)
    else:  # monthly
        locator = mdates.MonthLocator(interval=1)
    
    ax.xaxis.set_major_locator(locator)
    # Use YYYY-MM format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Set rotation and alignment to prevent overlap
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')
    ax.tick_params(labelsize=9)
    
    # Ensure bottom margin for rotated labels (update figure's bottom margin)
    fig = ax.figure
    current_bottom = getattr(fig, '_current_bottom', 0.11)
    fig._current_bottom = max(current_bottom, 0.15)


def apply_report_style(ax: Axes, title: str, subtitle: Optional[str] = None) -> None:
    """
    Apply consistent report styling to axes.
    
    Args:
        ax: Matplotlib axes
        title: Main title (bold, large)
        subtitle: Subtitle text (italic, smaller, directly under title)
    """
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        if subtitle:
            # Add subtitle as second line
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=10, style='italic', alpha=0.8)
    
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)


def add_source_footer(fig: Figure, sources: List[str], note: Optional[str] = None) -> None:
    """
    Add consistent source footer to figure.
    
    Args:
        fig: Matplotlib figure
        sources: List of source strings
        note: Optional note (e.g., "Macro FRED not loaded")
    """
    source_parts = ["Source:"] + sources
    if note:
        source_parts.append(f"Note: {note}")
    
    source_text = " ".join(source_parts)
    fig.text(0.01, 0.01, source_text, ha='left', va='bottom',
            fontsize=8, style='italic', alpha=0.7, transform=fig.transFigure)
    
    # Reserve space at bottom
    fig.subplots_adjust(bottom=0.12)


def finalize_figure(fig: Figure, *, title: str, subtitle: Optional[str] = None, 
                   source_text: Optional[str] = None, top: float = 0.90, 
                   bottom: float = 0.12) -> None:
    """
    Finalize figure with title, subtitle, source, and proper spacing.
    Ensures no clipping of any elements.
    
    Args:
        fig: Matplotlib figure
        title: Main title
        subtitle: Optional subtitle
        source_text: Optional source text
        top: Top margin (default 0.90)
        bottom: Bottom margin (default 0.12)
    """
    # Add title and subtitle
    add_title_subtitle(fig, title, subtitle)
    
    # Add source if provided
    if source_text:
        add_source(fig, source_text)
    
    # Ensure proper margins
    fig.subplots_adjust(top=top, bottom=bottom, left=0.10, right=0.95)
    
    # Set white background
    fig.patch.set_facecolor('white')


def risk_off_spans(df: pd.DataFrame, date_col: str = "week_start", 
                   label_col: str = "regime_label") -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return merged contiguous (start, end) spans for RISK_OFF periods.
    
    Args:
        df: DataFrame with date_col and label_col
        date_col: Column name for dates
        label_col: Column name for regime labels
    
    Returns:
        List of (start_date, end_date) tuples for RISK_OFF spans
    """
    if df.empty:
        return []
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    spans = []
    in_risk_off = False
    block_start = None
    
    for idx, row in df.iterrows():
        is_risk_off = row[label_col] == "RISK_OFF"
        
        if is_risk_off and not in_risk_off:
            # Start of block
            block_start = row[date_col]
            in_risk_off = True
        elif not is_risk_off and in_risk_off:
            # End of block
            block_end = row[date_col]
            spans.append((block_start, block_end))
            in_risk_off = False
    
    # Handle block that extends to end
    if in_risk_off and block_start is not None:
        block_end = df[date_col].iloc[-1]
        spans.append((block_start, block_end))
    
    return spans

