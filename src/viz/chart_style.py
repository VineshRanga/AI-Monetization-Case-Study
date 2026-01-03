"""Reusable chart styling utilities for director-level quant quality."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional

# Global constants
FOOTNOTE_TEXT = "Source: SEC EDGAR (fundamentals), Stooq (equities). Macro FRED not loaded. Regime: QQQ/SMH fallback."
DPI = 220
GRID_ALPHA = 0.25
TITLE_FONTSIZE = 15
SUBTITLE_FONTSIZE = 11
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 10
FOOTNOTE_FONTSIZE = 8


def apply_style(
    ax: Axes,
    title: str,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    footnote: Optional[str] = None,
) -> None:
    """
    Apply consistent styling to an axes object.
    
    Args:
        ax: Matplotlib axes object
        title: Main title (fontsize 15, bold)
        subtitle: Subtitle text (fontsize 11, italic, centered above title)
        xlabel: X-axis label (fontsize 11, bold)
        ylabel: Y-axis label (fontsize 11, bold)
        footnote: Footnote text (fontsize 8, italic, bottom-left)
    """
    # Title
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    
    # Subtitle (if provided)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
               fontsize=SUBTITLE_FONTSIZE, ha='center', style='italic')
    
    # Axis labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=GRID_ALPHA)
    
    # Tick fontsize
    ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # Footnote (if provided)
    if footnote:
        add_footnote(ax, footnote)


def add_footnote(ax: Axes, text: str) -> None:
    """Add footnote text to bottom-left of axes."""
    ax.text(0.02, 0.02, text, transform=ax.transAxes, 
           fontsize=FOOTNOTE_FONTSIZE, style='italic', alpha=0.7,
           verticalalignment='bottom', horizontalalignment='left')


def format_date_axis(ax: Axes, rotation: int = 35) -> None:
    """
    Format date axis with monthly ticks and rotation.
    
    Args:
        ax: Matplotlib axes object
        rotation: Rotation angle for date labels (default 35)
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Minor ticks weekly (optional, for cleaner charts)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')
    ax.tick_params(labelsize=TICK_FONTSIZE)


def add_metrics_inset(
    ax: Axes,
    metrics: dict,
    position: tuple = (0.98, 0.98),
    ha: str = 'right',
    va: str = 'top',
) -> None:
    """
    Add metrics text box to axes.
    
    Args:
        ax: Matplotlib axes object
        metrics: Dictionary of metric name -> value
        position: (x, y) position in axes coordinates (default top-right)
        ha: Horizontal alignment (default 'right')
        va: Vertical alignment (default 'top')
    """
    # Format metrics text
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01:
                lines.append(f"{key}: {value:.4e}")
            else:
                lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    
    textstr = '\n'.join(lines)
    ax.text(position[0], position[1], textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment=va, horizontalalignment=ha,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def shade_regime_periods(
    ax: Axes,
    regime_df,
    week_col: str = 'week_start',
    regime_col: str = 'regime_label',
    regime_value: str = 'RISK_OFF',
    alpha: float = 0.2,
    color: str = 'red',
    label: Optional[str] = None,
) -> None:
    """
    Shade periods where regime equals regime_value.
    
    Args:
        ax: Matplotlib axes object
        regime_df: DataFrame with week_start and regime_label columns
        week_col: Column name for week start dates
        regime_col: Column name for regime labels
        regime_value: Value to shade (default 'RISK_OFF')
        alpha: Transparency (default 0.2)
        color: Shading color (default 'red')
        label: Label for legend (default None, uses regime_value)
    """
    import pandas as pd
    
    if regime_df.empty:
        return
    
    # Ensure week_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(regime_df[week_col]):
        regime_df = regime_df.copy()
        regime_df[week_col] = pd.to_datetime(regime_df[week_col])
    
    shaded = False
    for idx, row in regime_df.iterrows():
        if row[regime_col] == regime_value:
            # Get next week for shading extent
            if idx < len(regime_df) - 1:
                next_week = regime_df.iloc[idx + 1][week_col]
            else:
                # Last row: extend to same week (single point shading)
                next_week = row[week_col]
            
            ax.axvspan(
                row[week_col],
                next_week,
                alpha=alpha,
                color=color,
                label=label if not shaded and label else (regime_value if not shaded else "")
            )
            shaded = True

