"""Shared plotting style helper for director-level quant quality."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional

# Global constants
FOOTNOTE_TEXT = "Source: SEC EDGAR (fundamentals), Stooq (equities). Macro FRED not loaded. Regime: QQQ/SMH fallback."
DPI = 200


def apply_mpl_style():
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
    })


def add_title_subtitle(fig: Figure, title: str, subtitle: Optional[str] = None) -> None:
    """
    Add title and subtitle to figure using ONLY fig.suptitle (no ax.set_title).
    
    Args:
        fig: Matplotlib figure
        title: Main title (fontsize 13, bold)
        subtitle: Subtitle text (fontsize 10, italic, placed below title)
    """
    # Main title via suptitle
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Subtitle via fig.text (if provided)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', va='top', 
                fontsize=10, style='italic', alpha=0.8)
    
    # Reserve space at top
    fig.subplots_adjust(top=0.88)


def add_source_footnote(fig: Figure, text: str) -> None:
    """
    Add source footnote OUTSIDE axes area.
    
    Args:
        fig: Matplotlib figure
        text: Footnote text (fontsize 8, italic)
    """
    fig.text(0.01, 0.01, text, ha='left', va='bottom',
            fontsize=8, style='italic', alpha=0.7)
    
    # Reserve space at bottom
    fig.subplots_adjust(bottom=0.12)


def save_fig(fig: Figure, path: Path) -> None:
    """
    Save figure with consistent settings and close.
    
    Args:
        fig: Matplotlib figure
        path: Output path
    """
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def format_date_axis_robust(ax, rotation: int = 45, interval_months: int = 3):
    """
    Format date axis with readable monthly ticks (not weekly spam).
    
    Args:
        ax: Matplotlib axes
        rotation: Rotation angle for labels (default 45)
        interval_months: Major tick interval in months (default 3)
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')
    ax.tick_params(labelsize=9)


def shade_regime_contiguous(ax, regime_df, week_col: str = 'week_start',
                            regime_col: str = 'regime_label', regime_value: str = 'RISK_OFF',
                            alpha: float = 0.2, color: str = 'red'):
    """
    Shade RISK_OFF periods as contiguous blocks (not per-week stripes).
    
    Args:
        ax: Matplotlib axes
        regime_df: DataFrame with week_start and regime_label
        week_col: Column name for dates
        regime_col: Column name for regime labels
        regime_value: Value to shade
        alpha: Transparency
        color: Shading color
    """
    import pandas as pd
    
    if regime_df.empty:
        return
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(regime_df[week_col]):
        regime_df = regime_df.copy()
        regime_df[week_col] = pd.to_datetime(regime_df[week_col])
    
    # Sort by date
    regime_df = regime_df.sort_values(week_col).reset_index(drop=True)
    
    # Find contiguous RISK_OFF blocks
    in_risk_off = False
    block_start = None
    
    for idx, row in regime_df.iterrows():
        is_risk_off = row[regime_col] == regime_value
        
        if is_risk_off and not in_risk_off:
            # Start of block
            block_start = row[week_col]
            in_risk_off = True
        elif not is_risk_off and in_risk_off:
            # End of block
            block_end = row[week_col]
            ax.axvspan(block_start, block_end, alpha=alpha, color=color, zorder=0)
            in_risk_off = False
    
    # Handle block that extends to end
    if in_risk_off and block_start is not None:
        block_end = regime_df[week_col].iloc[-1]
        ax.axvspan(block_start, block_end, alpha=alpha, color=color, zorder=0)

