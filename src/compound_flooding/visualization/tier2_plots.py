"""
Tier 2 visualization module for compound flooding analysis.

This module provides visualizations for Tier 2 analysis results, including:
- Copula density visualizations
- Tail dependence analysis
- Joint return period contours
- Conditional exceedance probabilities
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import base visualization utilities
from src.compound_flooding.visualization.base import (
    FIG_SIZES, set_publication_style, save_figure, 
    RED_BLUE_CMAP, CPR_CMAP, RISK_CMAP, SEA_CMAP, PRECIP_CMAP
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_copula_density(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a visualization of the fitted copula density.
    
    Parameters
    ----------
    station_data : Dict
        Tier 2 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if copula data is available
    if ('tier2_analysis' not in station_data or 
        'copula' not in station_data['tier2_analysis']):
        logger.warning("No copula data available")
        return None
    
    # Extract copula information
    copula_info = station_data['tier2_analysis']['copula']
    copula_method = copula_info.get('method', 'unknown')
    params = copula_info.get('parameters', {})
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
    station_code = station_data.get('station', '')
    
    # Create a grid of points for density evaluation
    u = np.linspace(0.01, 0.99, 100)
    v = np.linspace(0.01, 0.99, 100)
    uu, vv = np.meshgrid(u, v)
    
    # Compute copula density based on the method and parameters
    density = np.ones((100, 100))  # Default to independence
    
    if copula_method == 'Gaussian':
        # For Gaussian copula
        rho = params.get('rho', 0)
        if rho:
            # Generate density from a bivariate Gaussian
            from scipy.stats import norm, multivariate_normal
            
            # Convert from uniform to normal scores
            norm_u = norm.ppf(uu)
            norm_v = norm.ppf(vv)
            
            # Compute bivariate normal density
            cov = np.array([[1, rho], [rho, 1]])
            mvn = multivariate_normal(mean=[0, 0], cov=cov)
            
            # Stack coordinates for evaluation
            coords = np.dstack((norm_u, norm_v))
            density = mvn.pdf(coords)
            
            # Normalize density to [0, 1] range for visualization
            density = (density - density.min()) / (density.max() - density.min())
    
    elif copula_method == 'Gumbel':
        # For Gumbel copula
        theta = params.get('theta', 1)
        if theta > 1:
            # Compute Gumbel density (approximation for visualization)
            # This is simplified and not exact
            w = -np.log(uu) + -np.log(vv)
            w_theta = w**theta
            log_density = (
                np.log(theta) + (theta-1)*np.log(w) - w_theta +
                (theta-1)*np.log(-np.log(uu)) + (theta-1)*np.log(-np.log(vv)) +
                np.log(uu) + np.log(vv) + 
                w_theta * (1 + (theta-1)/w)
            )
            density = np.exp(log_density)
            # Handle numerical issues
            density[~np.isfinite(density)] = 0
            if density.max() > 0:
                density = (density - density.min()) / (density.max() - density.min())
    
    elif copula_method == 'Frank':
        # For Frank copula
        theta = params.get('theta', 0)
        if theta != 0:
            # Compute Frank density (approximation for visualization)
            numerator = -theta * np.exp(-theta * (uu + vv))
            denominator = (1 - np.exp(-theta)) - (1 - np.exp(-theta*uu))*(1 - np.exp(-theta*vv))
            density = numerator / denominator**2
            # Handle numerical issues
            density[~np.isfinite(density)] = 0
            if density.max() > 0:
                density = (density - density.min()) / (density.max() - density.min())
    
    elif copula_method == 'StudentT':
        # For Student T copula
        rho = params.get('rho', 0)
        df = params.get('df', 4)
        if rho:
            # This is a simplified implementation for visualization
            from scipy.stats import norm, t
            
            # Convert from uniform to t scores
            t_u = t.ppf(uu, df)
            t_v = t.ppf(vv, df)
            
            # Compute approximate density (this is not exact)
            density = np.exp(-0.5 * (t_u**2 + t_v**2 - 2*rho*t_u*t_v) / (1-rho**2))
            # Handle numerical issues
            density[~np.isfinite(density)] = 0
            if density.max() > 0:
                density = (density - density.min()) / (density.max() - density.min())
    
    elif copula_method == 'Clayton':
        # For Clayton copula
        theta = params.get('theta', 0)
        if theta > 0:
            # Compute Clayton density (approximation for visualization)
            density = np.zeros((100, 100))
            for i in range(100):
                for j in range(100):
                    ui, vj = uu[i,j], vv[i,j]
                    if ui <= 0 or vj <= 0:
                        continue
                    term = ui**(-theta) + vj**(-theta) - 1
                    if term <= 0:
                        continue
                    density[i,j] = (1+theta) * (ui*vj)**(-1-theta) * term**(-(2+1/theta))
                
            # Handle numerical issues
            density[~np.isfinite(density)] = 0
            if density.max() > 0:
                density = (density - density.min()) / (density.max() - density.min())
    
    # Create contour plot
    contour = ax.contourf(uu, vv, density, levels=20, cmap=RED_BLUE_CMAP)
    plt.colorbar(contour, ax=ax, label='Density')
    
    # Add contour lines
    ax.contour(uu, vv, density, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    
    # Add joint exceedance contours if available
    if ('joint_exceedance' in station_data['tier2_analysis'] and 
        '0.95' in station_data['tier2_analysis']['joint_exceedance']):
        
        # Mark 95% level
        ax.axvline(0.95, linestyle='--', color='blue', alpha=0.7)
        ax.axhline(0.95, linestyle='--', color='blue', alpha=0.7)
        
        # Add rectangle for joint exceedance region
        rect = patches.Rectangle((0.95, 0.95), 0.05, 0.05, 
                               edgecolor='red', facecolor='none', 
                               linestyle='-', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Add text with CPR value
        cpr_95 = station_data['tier2_analysis']['joint_exceedance']['0.95'].get('cpr', np.nan)
        if not np.isnan(cpr_95):
            ax.text(0.96, 0.96, f"CPR = {cpr_95:.2f}", 
                    fontsize=9, color='red', ha='left', va='bottom')
    
    # Labels and title
    ax.set_xlabel('Sea Level (uniform margin)')
    ax.set_ylabel('Precipitation (uniform margin)')
    ax.set_title(f"Copula Density - {station_code} - {copula_method}")
    
    # Add text with copula parameters
    if params:
        param_text = "Parameters:\n"
        for k, v in params.items():
            param_text += f"{k} = {v:.3f}\n"
        
        # Add information about tail dependence if available
        if 'tail_dependence' in station_data['tier2_analysis']:
            tail = station_data['tier2_analysis']['tail_dependence']
            param_text += f"\nTail Dependence:\n"
            param_text += f"Lower: {tail.get('lower', 'N/A'):.3f}\n"
            param_text += f"Upper: {tail.get('upper', 'N/A'):.3f}"
            
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_copula_density")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_joint_return_periods(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a visualization of joint return periods.
    
    Parameters
    ----------
    station_data : Dict
        Tier 2 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if joint return period data is available
    if ('tier2_analysis' not in station_data or 
        'joint_return_periods' not in station_data['tier2_analysis']):
        logger.warning("No joint return period data available")
        return None
    
    # Extract joint return period information
    joint_rp = station_data['tier2_analysis']['joint_return_periods']
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZES['large'])
    station_code = station_data.get('station', '')
    
    # Create a grid for visualization (in probability space)
    u = np.linspace(0.01, 0.99, 100)
    v = np.linspace(0.01, 0.99, 100)
    uu, vv = np.meshgrid(u, v)
    
    # Create array for joint return periods
    and_rp = np.ones((100, 100))  # Default to 1-year return period
    
    # Extract return period information for specific levels
    rp_data = {}
    for rp_key, rp_val in joint_rp.items():
        if isinstance(rp_val, dict) and 'and_return_period' in rp_val:
            try:
                rp_num = float(rp_key)
                u_level = rp_val.get('u_level', np.nan)
                v_level = rp_val.get('v_level', np.nan)
                and_rp_val = rp_val.get('and_return_period', np.nan)
                
                if not np.isnan(u_level) and not np.isnan(v_level) and not np.isnan(and_rp_val):
                    rp_data[rp_num] = {
                        'u_level': u_level,
                        'v_level': v_level,
                        'and_rp': and_rp_val
                    }
            except ValueError:
                # Skip if key is not a number
                pass
    
    # If we have enough data points, create a smooth contour plot
    if len(rp_data) >= 3:
        # Extract data for contour plot
        levels = np.array([0.9, 0.95, 0.99])  # Example fixed probability levels
        
        # We need to simulate the full joint return period surface
        # Extract copula information
        if ('copula' in station_data['tier2_analysis'] and
            'method' in station_data['tier2_analysis']['copula'] and
            'parameters' in station_data['tier2_analysis']['copula']):
            
            copula_method = station_data['tier2_analysis']['copula']['method']
            params = station_data['tier2_analysis']['copula']['parameters']
            
            # Approximate joint exceedance probability P(X>x, Y>y)
            # For different copula types
            joint_exc = np.zeros((100, 100))
            
            if copula_method == 'Gaussian':
                rho = params.get('rho', 0)
                # Crude approximation for Gaussian copula
                for i in range(100):
                    for j in range(100):
                        ui, vj = uu[i,j], vv[i,j]
                        p_i = 1 - ui
                        p_j = 1 - vj
                        p_ind = p_i * p_j
                        r_factor = min(3.0, max(0.1, 1 + rho))  # Constrain to reasonable values
                        joint_exc[i,j] = min(p_i, p_j) * r_factor
            
            elif copula_method == 'Gumbel':
                theta = params.get('theta', 1)
                # Crude approximation for Gumbel copula
                for i in range(100):
                    for j in range(100):
                        ui, vj = uu[i,j], vv[i,j]
                        p_i = 1 - ui
                        p_j = 1 - vj
                        p_ind = p_i * p_j
                        # Stronger upper tail dependence for theta > 1
                        if ui > 0.9 and vj > 0.9:
                            r_factor = min(5.0, max(0.1, theta))
                        else:
                            r_factor = min(3.0, max(0.5, 1 + (theta-1)/2))
                        joint_exc[i,j] = min(p_i, p_j) * r_factor
            
            else:
                # For other copula types use a generic approximation
                # based on return period data we have
                for rp_num, data in rp_data.items():
                    u_level = data['u_level']
                    v_level = data['v_level']
                    and_rp_val = data['and_rp']
                    
                    # Convert return period back to exceedance probability
                    p_joint = 1 / and_rp_val
                    
                    # Use this to set a specific point in our grid
                    u_idx = int(u_level * 99)
                    v_idx = int(v_level * 99)
                    joint_exc[u_idx, v_idx] = p_joint
                
                # Interpolate/extrapolate to fill the grid
                # This is very approximate
                from scipy.interpolate import griddata
                
                # Create sparse points from known values
                points = []
                values = []
                for i in range(100):
                    for j in range(100):
                        if joint_exc[i,j] > 0:
                            points.append([i, j])
                            values.append(joint_exc[i,j])
                
                # Add boundary conditions
                for i in range(100):
                    # P(X>x,Y>0) = P(X>x)
                    points.append([i, 0])
                    values.append(1-uu[i,0])
                    
                    # P(X>0,Y>y) = P(Y>y)
                    points.append([0, i])
                    values.append(1-vv[0,i])
                
                # If we have enough points, interpolate
                if len(points) > 3:
                    grid_x, grid_y = np.mgrid[0:100, 0:100]
                    joint_exc = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0)
            
            # Convert joint exceedance probability to return period
            # T = 1/p
            with np.errstate(divide='ignore'):
                and_rp = 1 / joint_exc
                and_rp[~np.isfinite(and_rp)] = 1000  # Cap at 1000 years
                and_rp[and_rp < 1] = 1  # Minimum 1-year return period
        
        # Define return period levels for contours
        rp_levels = [2, 5, 10, 20, 50, 100, 200, 500]
        
        # Create filled contour plot
        contour = ax.contourf(uu, vv, and_rp, levels=rp_levels, 
                             norm='log', cmap=RISK_CMAP)
        plt.colorbar(contour, ax=ax, label='Joint Return Period (years)')
        
        # Add contour lines with labels
        cs = ax.contour(uu, vv, and_rp, levels=rp_levels, 
                       colors='black', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
        
    # If we don't have enough data for a contour plot, create a scatter plot
    # of the known return periods
    else:
        # Create a colormap for the scatter points
        min_rp = min([d['and_rp'] for d in rp_data.values()])
        max_rp = max([d['and_rp'] for d in rp_data.values()])
        
        # Plot points with text labels
        for rp_num, data in rp_data.items():
            ax.scatter(data['u_level'], data['v_level'], 
                      c=[data['and_rp']], cmap=RISK_CMAP, 
                      norm='log', vmin=min_rp, vmax=max_rp, s=100, zorder=10)
            
            ax.text(data['u_level'], data['v_level'], f"{data['and_rp']:.0f}yr", 
                   fontsize=9, ha='center', va='bottom', zorder=11)
        
        # Add a colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm
        sm = ScalarMappable(cmap=RISK_CMAP, norm=LogNorm(vmin=min_rp, vmax=max_rp))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Joint Return Period (years)')
    
    # Add reference lines for independence case
    for t in [10, 100]:
        # T = 1/p, so p = 1/T
        p = 1/t
        level = 1-p
        
        # Draw lines for:
        # In independent case, P(X>x,Y>y) = P(X>x)*P(Y>y) = p*p = p^2
        # So joint exceedance level is 1 - p^2
        label = f"T={t}yr (independent)"
        
        # Mark individual variable return period levels
        ax.axvline(level, linestyle=':', color='blue', alpha=0.5)
        ax.axhline(level, linestyle=':', color='blue', alpha=0.5)
        
        # Add text labels
        ax.text(level, 0.05, f"{t}yr", color='blue', fontsize=8, ha='center')
        ax.text(0.05, level, f"{t}yr", color='blue', fontsize=8, va='center')
    
    # Labels and title
    ax.set_xlabel('Sea Level (uniform margin)')
    ax.set_ylabel('Precipitation (uniform margin)')
    ax.set_title(f"Joint Return Periods - {station_code}")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_joint_return_periods")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_conditional_exceedance(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a visualization of conditional exceedance probabilities.
    
    Parameters
    ----------
    station_data : Dict
        Tier 2 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if conditional probability data is available
    if ('tier2_analysis' not in station_data or 
        'conditional_probabilities' not in station_data['tier2_analysis']):
        logger.warning("No conditional probability data available")
        return None
    
    # Extract conditional probability information
    cond_probs = station_data['tier2_analysis']['conditional_probabilities']
    
    # Check if we have at least one level of data
    if not cond_probs or not isinstance(cond_probs, dict) or len(cond_probs) == 0:
        logger.warning("Conditional probability data is empty or not a dictionary")
        return None
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['wide'])
    station_code = station_data.get('station', '')
    
    # Prepare data for plotting
    levels = []
    p_v_given_u_med = []
    p_u_given_v_med = []
    p_v_given_u_exceed = []
    p_u_given_v_exceed = []
    
    for level, data in cond_probs.items():
        try:
            level_float = float(level)
            levels.append(level_float)
            p_v_given_u_med.append(data.get('p_v_given_u_med', np.nan))
            p_u_given_v_med.append(data.get('p_u_given_v_med', np.nan))
            p_v_given_u_exceed.append(data.get('p_v_given_u_exceed', np.nan))
            p_u_given_v_exceed.append(data.get('p_u_given_v_exceed', np.nan))
        except (ValueError, TypeError):
            # Skip if level can't be converted to float
            pass
    
    # Skip if we don't have enough data
    if len(levels) < 2:
        logger.warning("Not enough conditional probability data for plotting")
        return None
    
    # Sort by level
    sort_idx = np.argsort(levels)
    levels = [levels[i] for i in sort_idx]
    p_v_given_u_med = [p_v_given_u_med[i] for i in sort_idx]
    p_u_given_v_med = [p_u_given_v_med[i] for i in sort_idx]
    p_v_given_u_exceed = [p_v_given_u_exceed[i] for i in sort_idx]
    p_u_given_v_exceed = [p_u_given_v_exceed[i] for i in sort_idx]
    
    # PLOT 1: Conditional given median
    ax = axes[0]
    
    # Plot P(V>v | U=u_med) and P(U>u | V=v_med)
    ax.plot(levels, p_v_given_u_med, 'o-', color='blue', 
            label='P(PR>pr | SL=median)')
    ax.plot(levels, p_u_given_v_med, 's-', color='green',
            label='P(SL>sl | PR=median)')
    
    # Add reference line for unconditional probability
    ax.plot(levels, 1 - np.array(levels), 'k--', label='Unconditional')
    
    # Labels and formatting
    ax.set_xlabel('Threshold Level')
    ax.set_ylabel('Conditional Probability')
    ax.set_title('Conditional on Median')
    ax.set_xlim(min(levels), max(levels))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # PLOT 2: Conditional given exceedance
    ax = axes[1]
    
    # Plot P(V>v | U>u) and P(U>u | V>v)
    ax.plot(levels, p_v_given_u_exceed, 'o-', color='blue',
            label='P(PR>pr | SL>sl)')
    ax.plot(levels, p_u_given_v_exceed, 's-', color='green',
            label='P(SL>sl | PR>pr)')
    
    # Add reference line for unconditional probability
    ax.plot(levels, 1 - np.array(levels), 'k--', label='Unconditional')
    
    # Labels and formatting
    ax.set_xlabel('Threshold Level')
    ax.set_ylabel('Conditional Probability')
    ax.set_title('Conditional on Exceedance')
    ax.set_xlim(min(levels), max(levels))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Overall title
    fig.suptitle(f"Conditional Exceedance Probabilities - {station_code}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_conditional_exceedance")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_tail_dependence(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a visualization of tail dependence coefficients.
    
    Parameters
    ----------
    station_data : Dict
        Tier 2 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if tail dependence data is available
    if ('tier2_analysis' not in station_data or 
        'tail_dependence' not in station_data['tier2_analysis']):
        logger.warning("No tail dependence data available")
        return None
    
    # Extract tail dependence information
    tail_dep = station_data['tier2_analysis']['tail_dependence']
    lower_tail = tail_dep.get('lower', np.nan)
    upper_tail = tail_dep.get('upper', np.nan)
    
    # Skip if both are NaN
    if np.isnan(lower_tail) and np.isnan(upper_tail):
        logger.warning("Both tail dependence coefficients are NaN")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
    station_code = station_data.get('station', '')
    
    # Create a simple bar chart
    labels = ['Lower Tail', 'Upper Tail']
    values = [lower_tail, upper_tail]
    colors = ['blue', 'red']
    
    # Create the bars
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            height = value
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=12)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.02,
                   'N/A', ha='center', va='bottom', fontsize=12)
    
    # Add a reference line at zero (no tail dependence)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='No Tail Dependence')
    
    # Add reference labels for tail dependence interpretation
    dependence_levels = [
        (0.0, 'No Dependence'),
        (0.3, 'Moderate Dependence'),
        (0.6, 'Strong Dependence'),
        (0.9, 'Very Strong Dependence')
    ]
    
    for level, label in dependence_levels:
        ax.axhline(level, color='gray', linestyle=':', alpha=0.5)
        ax.text(1.8, level, label, fontsize=8, va='center')
    
    # Get copula information for additional context
    copula_info = None
    if ('copula' in station_data['tier2_analysis'] and 
        'method' in station_data['tier2_analysis']['copula']):
        copula_method = station_data['tier2_analysis']['copula']['method']
        params = station_data['tier2_analysis']['copula'].get('parameters', {})
        
        copula_info = f"Copula: {copula_method}"
        for k, v in params.items():
            copula_info += f", {k}={v:.3f}"
    
    # Add copula information text
    if copula_info:
        ax.text(0.5, 0.05, copula_info, fontsize=10, ha='center', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Labels and title
    ax.set_ylabel('Tail Dependence Coefficient')
    ax.set_title(f"Tail Dependence - {station_code}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_tail_dependence")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_station_tier2_summary(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive Tier 2 summary plot for a station.
    
    Parameters
    ----------
    station_data : Dict
        Tier 2 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES['full'])
    station_code = station_data.get('station', '')
    fig.suptitle(f"Tier 2 Analysis Summary - {station_code}", fontsize=16)
    
    # PLOT 1: Copula Density
    ax = axes[0, 0]
    
    # Check if copula data is available
    if ('tier2_analysis' in station_data and 
        'copula' in station_data['tier2_analysis']):
        
        # Extract copula information
        copula_info = station_data['tier2_analysis']['copula']
        copula_method = copula_info.get('method', 'unknown')
        params = copula_info.get('parameters', {})
        
        # Create a simplified copula density visualization
        u = np.linspace(0.01, 0.99, 50)  # reduced resolution for summary
        v = np.linspace(0.01, 0.99, 50)
        uu, vv = np.meshgrid(u, v)
        
        # Compute simplified density for visualization
        density = np.ones((50, 50))  # Default to independence
        
        if copula_method == 'Gaussian':
            # Simplified Gaussian case
            rho = params.get('rho', 0)
            if rho:
                from scipy.stats import multivariate_normal
                cov = np.array([[1, rho], [rho, 1]])
                mvn = multivariate_normal(mean=[0, 0], cov=cov)
                
                # Simplified density (no exact transform to uniform margins)
                xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
                pos = np.dstack((xx, yy))
                density = mvn.pdf(pos)
        
        # Create contour plot
        contour = ax.contourf(uu, vv, density, levels=10, cmap=RED_BLUE_CMAP)
        
        # Add parameter annotation
        if params:
            param_text = f"Copula: {copula_method}\n"
            for k, v in params.items():
                param_text += f"{k} = {v:.3f}\n"
                
            ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Copula Density')
    ax.set_xlabel('Sea Level (U)')
    ax.set_ylabel('Precipitation (V)')
    
    # PLOT 2: Joint Return Periods
    ax = axes[0, 1]
    
    # Check if joint return period data is available
    if ('tier2_analysis' in station_data and 
        'joint_return_periods' in station_data['tier2_analysis']):
        
        # Extract joint return period data
        joint_rp = station_data['tier2_analysis']['joint_return_periods']
        
        # Plot specific return period levels
        key_rps = ['10', '100']
        markers = ['o', 's']
        colors = ['blue', 'red']
        
        for key_rp, marker, color in zip(key_rps, markers, colors):
            if key_rp in joint_rp:
                rp_data = joint_rp[key_rp]
                u_level = rp_data.get('u_level', np.nan)
                v_level = rp_data.get('v_level', np.nan)
                and_rp = rp_data.get('and_return_period', np.nan)
                
                if not np.isnan(u_level) and not np.isnan(v_level) and not np.isnan(and_rp):
                    ax.scatter(u_level, v_level, s=100, color=color, marker=marker, zorder=10)
                    ax.text(u_level, v_level+0.05, f"{key_rp}yr → {and_rp:.0f}yr", 
                            fontsize=8, ha='center', va='bottom', color=color, zorder=11)
                    
                    # Draw lines to axes for reference
                    ax.plot([0, u_level], [v_level, v_level], 
                            color=color, linestyle=':', alpha=0.5)
                    ax.plot([u_level, u_level], [0, v_level], 
                            color=color, linestyle=':', alpha=0.5)
        
        # Create a grid for visualization
        u = np.linspace(0.01, 0.99, 50)
        v = np.linspace(0.01, 0.99, 50)
        uu, vv = np.meshgrid(u, v)
        
        # Plot return period isolines for reference
        # We'll use a simplistic model for visualization purposes
        levels = [2, 5, 10, 20, 50, 100, 200, 500]
        
        # Get representative and_rp values from the data
        rp_values = []
        u_values = []
        v_values = []
        
        for rp_key, rp_val in joint_rp.items():
            try:
                rp = float(rp_key)
                and_rp = rp_val.get('and_return_period', np.nan)
                u_level = rp_val.get('u_level', np.nan)
                v_level = rp_val.get('v_level', np.nan)
                
                if not np.isnan(and_rp) and not np.isnan(u_level) and not np.isnan(v_level):
                    rp_values.append(rp)
                    u_values.append(u_level)
                    v_values.append(v_level)
            except ValueError:
                pass
        
        # If we have at least one value, create a simple contour for visualization
        if len(rp_values) > 0:
            # Create a simple model for joint return periods
            # based on available data points
            # This is a very crude approximation for visualization
            
            # Compare the actual joint return period to the independence case
            independence_factor = 1.0
            if len(rp_values) > 0:
                # Get the highest return period
                max_idx = np.argmax(rp_values)
                rp = rp_values[max_idx]
                and_rp = joint_rp[str(rp)].get('and_return_period', rp)
                
                # In independence case, T_and = 1/(p^2) = T^2
                if and_rp < rp**2:
                    # Negative dependence
                    independence_factor = 0.5
                else:
                    # Positive dependence
                    independence_factor = 2.0
            
            # Generate a simplified model for visualization
            and_rp_grid = np.zeros((50, 50))
            for i in range(50):
                for j in range(50):
                    ui, vj = uu[i,j], vv[i,j]
                    p_i = 1 - ui
                    p_j = 1 - vj
                    p_joint = p_i * p_j * independence_factor
                    and_rp_grid[i,j] = 1 / max(p_joint, 1e-6)
            
            # Create contour lines
            cs = ax.contour(uu, vv, and_rp_grid, levels=levels, 
                           colors='black', linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%dyr')
    
    ax.set_title('Joint Return Periods')
    ax.set_xlabel('Sea Level (U)')
    ax.set_ylabel('Precipitation (V)')
    ax.grid(True, alpha=0.3)
    
    # PLOT 3: Tail Dependence
    ax = axes[1, 0]
    
    # Check if tail dependence data is available
    if ('tier2_analysis' in station_data and 
        'tail_dependence' in station_data['tier2_analysis']):
        
        # Extract tail dependence information
        tail_dep = station_data['tier2_analysis']['tail_dependence']
        lower_tail = tail_dep.get('lower', np.nan)
        upper_tail = tail_dep.get('upper', np.nan)
        
        if not np.isnan(lower_tail) or not np.isnan(upper_tail):
            # Create simple bar chart
            labels = ['Lower Tail', 'Upper Tail']
            values = [lower_tail, upper_tail]
            colors = ['blue', 'red']
            
            # Create bars
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = value
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., 0.02,
                           'N/A', ha='center', va='bottom', fontsize=10)
            
            # Add reference line for no tail dependence
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
    
    ax.set_title('Tail Dependence')
    ax.set_ylabel('Coefficient')
    ax.grid(True, alpha=0.3)
    
    # PLOT 4: Conditional Exceedance Probabilities
    ax = axes[1, 1]
    
    # Check if conditional probability data is available
    if ('tier2_analysis' in station_data and 
        'conditional_probabilities' in station_data['tier2_analysis']):
        
        # Extract conditional probability information
        cond_probs = station_data['tier2_analysis']['conditional_probabilities']
        
        # Prepare data for plotting
        levels = []
        p_v_given_u_exceed = []
        p_u_given_v_exceed = []
        
        for level, data in cond_probs.items():
            try:
                level_float = float(level)
                levels.append(level_float)
                p_v_given_u_exceed.append(data.get('p_v_given_u_exceed', np.nan))
                p_u_given_v_exceed.append(data.get('p_u_given_v_exceed', np.nan))
            except (ValueError, TypeError):
                pass
        
        # Plot if we have data
        if len(levels) >= 2:
            # Sort by level
            sort_idx = np.argsort(levels)
            levels = [levels[i] for i in sort_idx]
            p_v_given_u_exceed = [p_v_given_u_exceed[i] for i in sort_idx]
            p_u_given_v_exceed = [p_u_given_v_exceed[i] for i in sort_idx]
            
            # Plot conditional probabilities
            ax.plot(levels, p_v_given_u_exceed, 'o-', color='blue',
                    label='P(PR>pr | SL>sl)')
            ax.plot(levels, p_u_given_v_exceed, 's-', color='green',
                    label='P(SL>sl | PR>pr)')
            
            # Add reference line for unconditional probability
            ax.plot(levels, 1 - np.array(levels), 'k--', label='Unconditional')
            
            # Set axis limits
            ax.set_xlim(min(levels), max(levels))
            ax.set_ylim(0, 1)
            
            # Add legend
            ax.legend(fontsize=8)
    
    ax.set_title('Conditional Exceedance')
    ax.set_xlabel('Threshold Level')
    ax.set_ylabel('Probability')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_tier2_summary")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_tau_vs_cpr(
    tier1_data: Dict, 
    tier2_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot relationship between Kendall's tau and CPR across stations.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract tau and CPR for each station
    stations = []
    taus = []
    cprs = []
    
    # Gather data
    for station_code in tier1_data.keys():
        # Skip if station not in tier2_data
        if station_code not in tier2_data:
            continue
            
        t1_data = tier1_data[station_code]
        t2_data = tier2_data[station_code]
        
        # Get CPR from tier1 data
        cpr = None
        if ('joint' in t1_data and 
            'empirical' in t1_data['joint']):
            cpr = t1_data['joint']['empirical'].get('cpr', None)
        
        # Get tau from tier2 data
        tau = None
        if ('tier2_analysis' in t2_data and 
            'copula' in t2_data['tier2_analysis'] and
            'parameters' in t2_data['tier2_analysis']['copula']):
            
            # If Gaussian or Student-T copula, convert rho to tau
            params = t2_data['tier2_analysis']['copula']['parameters']
            method = t2_data['tier2_analysis']['copula'].get('method', '')
            
            if method in ['Gaussian', 'StudentT'] and 'rho' in params:
                rho = params['rho']
                # Convert to tau using sin formula
                tau = 2 * np.arcsin(rho) / np.pi
            elif method == 'Gumbel' and 'theta' in params:
                theta = params['theta']
                # Tau = (theta-1)/theta for Gumbel
                tau = max(0, (theta - 1) / theta)
            elif method == 'Frank' and 'theta' in params:
                # For Frank, use Spearman's rho as proxy (approximate)
                theta = params['theta']
                if abs(theta) < 1e-6:
                    tau = 0
                else:
                    # Very rough approximation
                    tau = np.sign(theta) * min(0.8, abs(theta) / 10)
        
        # Add to lists if both are available
        if cpr is not None and tau is not None:
            stations.append(station_code)
            taus.append(tau)
            cprs.append(cpr)
    
    # Skip if we don't have enough data
    if len(stations) < 3:
        logger.warning("Not enough stations with both tau and CPR data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
    
    # Create scatter plot
    sc = ax.scatter(taus, cprs, c=cprs, cmap=CPR_CMAP, 
                   alpha=0.7, s=100, edgecolor='black')
    
    # Add colorbar
    plt.colorbar(sc, ax=ax, label='CPR')
    
    # Add station labels
    for i, station in enumerate(stations):
        ax.annotate(station, (taus[i], cprs[i]), 
                   fontsize=8, ha='right', va='bottom',
                   xytext=(5, 5), textcoords='offset points')
    
    # Fit a trend line
    if len(taus) >= 3:
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(taus, cprs)
            
            # Plot trend line
            x_range = np.linspace(min(taus), max(taus), 100)
            y_range = slope * x_range + intercept
            ax.plot(x_range, y_range, 'r--', 
                   label=f'y = {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.2f})')
            
            # Add statistics text
            stats_text = (f"Correlation: r = {r_value:.3f}\n"
                         f"p-value: {p_value:.4f}\n"
                         f"Slope: {slope:.3f} ± {std_err:.3f}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend
            ax.legend()
        except Exception as e:
            logger.warning(f"Error fitting trend line: {e}")
    
    # Add reference lines
    ax.axhline(1, color='black', linestyle='--', alpha=0.5, label='Independence (CPR=1)')
    ax.axvline(0, color='black', linestyle=':', alpha=0.5, label='Independence (τ=0)')
    
    # Labels and title
    ax.set_xlabel("Kendall's tau (τ)")
    ax.set_ylabel('Conditional Probability Ratio (CPR)')
    ax.set_title('Relationship between τ and CPR across Stations')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, "tau_vs_cpr_relationship")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def create_tier2_summary_report(
    tier1_data: Dict,
    tier2_data: Dict,
    output_dir: str,
    station_codes: List[str] = None,
    show: bool = False
) -> None:
    """
    Create comprehensive Tier 2 summary report with multiple visualizations.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    output_dir : str
        Directory to save figures
    station_codes : List[str], optional
        List of station codes to include. If None, include all stations.
    show : bool, optional
        Whether to display the figures
    """
    # Set publication style
    set_publication_style()
    
    # Filter stations if needed
    if station_codes:
        filtered_t2_data = {k: v for k, v in tier2_data.items() if k in station_codes}
        filtered_t1_data = {k: v for k, v in tier1_data.items() if k in station_codes}
    else:
        filtered_t2_data = tier2_data
        filtered_t1_data = tier1_data
        station_codes = list(tier2_data.keys())
    
    logger.info(f"Creating Tier 2 summary report for {len(filtered_t2_data)} stations")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'stations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'copulas'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dependence'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'joint_returns'), exist_ok=True)
    
    # Process each station
    for station_code, data in filtered_t2_data.items():
        logger.info(f"Generating Tier 2 visualizations for station {station_code}")
        
        # Copula density plot
        plot_copula_density(
            data,
            output_dir=os.path.join(output_dir, 'copulas'),
            show=show
        )
        
        # Joint return periods
        plot_joint_return_periods(
            data,
            output_dir=os.path.join(output_dir, 'joint_returns'),
            show=show
        )
        
        # Conditional exceedance
        plot_conditional_exceedance(
            data,
            output_dir=os.path.join(output_dir, 'dependence'),
            show=show
        )
        
        # Tail dependence
        plot_tail_dependence(
            data,
            output_dir=os.path.join(output_dir, 'dependence'),
            show=show
        )
        
        # Station summary
        plot_station_tier2_summary(
            data,
            output_dir=os.path.join(output_dir, 'stations'),
            show=show
        )
    
    # Create cross-station comparisons
    plot_tau_vs_cpr(
        filtered_t1_data, 
        filtered_t2_data,
        output_dir=output_dir,
        show=show
    )
    
    logger.info(f"Tier 2 summary report completed and saved to {output_dir}")


if __name__ == "__main__":
    # Basic test of the module
    import sys
    from src.compound_flooding.visualization.base import load_tier1_results, load_tier2_results, create_output_dirs
    
    print("Testing tier2_plots module...")
    
    # Set the style
    set_publication_style()
    
    # Check if we have a test data file
    if len(sys.argv) > 1:
        test_data_file = sys.argv[1]
        try:
            import json
            with open(test_data_file, 'r') as f:
                test_data = {os.path.splitext(os.path.basename(test_data_file))[0]: json.load(f)}
            
            # Create output directories
            dirs = create_output_dirs('outputs/plots_test')
            
            # Test the plotting functions
            station_code = list(test_data.keys())[0]
            station_data = test_data[station_code]
            
            plot_copula_density(station_data, dirs['tier2_copulas'])
            plot_joint_return_periods(station_data, dirs['tier2_joint_returns'])
            plot_conditional_exceedance(station_data, dirs['tier2_dependence'])
            plot_tail_dependence(station_data, dirs['tier2_dependence'])
            plot_station_tier2_summary(station_data, dirs['tier2_stations'])
            
            print(f"Test complete. Check outputs in {dirs['tier2']}")
        except Exception as e:
            print(f"Error testing with provided file: {e}")
    else:
        print("No test data provided. Please provide a Tier 2 output JSON file as argument.")