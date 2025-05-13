"""
Tier 3 visualization module for compound flooding analysis.

This module provides visualizations for Tier 3 Bayesian Network results, including:
- Network structure visualization
- Conditional probability tables
- Sensitivity analysis results
- Risk distribution analysis
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.backends.backend_agg  # Ensure backends are loaded
import seaborn as sns
import networkx as nx
import json
import glob
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import base visualization utilities
from src.compound_flooding.visualization.base import (
    FIG_SIZES, set_publication_style, save_figure, 
    RED_BLUE_CMAP, CPR_CMAP, RISK_CMAP, SEA_CMAP, PRECIP_CMAP
)

# Define node colors for different variable types
NODE_COLORS = {
    'Season': '#1f77b4',  # blue
    'Pressure_Anomaly': '#ff7f0e',  # orange
    'Wind_Speed': '#2ca02c',  # green
    'Wind_Direction': '#d62728',  # red
    'Sea_Level_State': '#9467bd',  # purple
    'Precipitation_State': '#8c564b',  # brown
    'Compound_Flood_Risk': '#e377c2',  # pink
}

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tier3_results(output_dir: str, station_code: str = None) -> Dict:
    """
    Load Tier 3 analysis results from JSON files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing Tier 3 output files (summary subdirectory)
    station_code : str, optional
        Specific station code to load. If None, load all stations.
        
    Returns
    -------
    Dict
        Dictionary of Tier 3 results, keyed by station code
    """
    results = {}
    
    # Define the summary directory
    summary_dir = os.path.join(output_dir, 'summary')
    if not os.path.exists(summary_dir):
        summary_dir = output_dir  # Try directly in output_dir if no summary subdirectory
    
    if station_code:
        # Look for a specific station
        json_file = os.path.join(summary_dir, f"{station_code}_tier3_bn.json")
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[station_code] = json.load(f)
        else:
            logger.warning(f"No Tier 3 results found for station {station_code}")
    else:
        # Load all stations
        json_files = glob.glob(os.path.join(summary_dir, "*_tier3_bn.json"))
        
        # Process JSON files
        for json_file in json_files:
            station_code = os.path.basename(json_file).split('_tier3_bn.json')[0]
            try:
                with open(json_file, 'r') as f:
                    results[station_code] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    
    return results


def plot_bayesian_network(
    station_data: Dict,
    ax: Optional[plt.Axes] = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the Bayesian Network structure.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract Bayesian Network structure
    if ('tier3_analysis' not in station_data or 
        'bayesian_network' not in station_data['tier3_analysis'] or
        'structure' not in station_data['tier3_analysis']['bayesian_network']):
        logger.warning("No Bayesian Network structure available")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
            ax.text(0.5, 0.5, "No Bayesian Network structure available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No Bayesian Network structure available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract structure information
    bn_structure = station_data['tier3_analysis']['bayesian_network']['structure']
    nodes = bn_structure.get('nodes', [])
    edges = bn_structure.get('edges', [])
    
    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
    else:
        fig = ax.figure
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Add edges
    for edge in edges:
        if isinstance(edge, list) and len(edge) == 2:
            G.add_edge(edge[0], edge[1])
    
    # Create a hierarchical layout (used for BNs to show causality)
    try:
        # Try using a hierarchical layout first
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fall back to spring layout if graphviz not available
        pos = nx.spring_layout(G, seed=42)
    
    # Extract sensitivity data if available
    sensitivity_data = {}
    if ('sensitivity' in station_data['tier3_analysis']['bayesian_network'] and
        isinstance(station_data['tier3_analysis']['bayesian_network']['sensitivity'], dict)):
        
        sensitivity_data = station_data['tier3_analysis']['bayesian_network']['sensitivity']
    
    # Create a dict mapping from node name to node size based on importance
    node_sizes = {}
    for node in nodes:
        if node == 'Compound_Flood_Risk':
            # Make the target node larger
            node_sizes[node] = 2000
        elif node in sensitivity_data:
            # Scale node size by average impact
            avg_impact = sensitivity_data[node].get('average_impact', 0)
            node_sizes[node] = 1000 + 5000 * avg_impact
        else:
            node_sizes[node] = 1000
    
    # Draw the network
    nx.draw_networkx_nodes(
        G, pos, 
        ax=ax,
        node_size=[node_sizes.get(node, 1000) for node in G.nodes()],
        node_color=[NODE_COLORS.get(node, '#aaaaaa') for node in G.nodes()],
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, pos, 
        ax=ax,
        arrows=True,
        arrowsize=20,
        edge_color='black',
        width=1.5,
        alpha=0.7
    )
    
    nx.draw_networkx_labels(
        G, pos, 
        ax=ax,
        font_size=12,
        font_weight='bold',
        font_color='black'
    )
    
    # Add a title
    station_code = station_data.get('station', '')
    ax.set_title(f"Bayesian Network Structure - {station_code}")
    
    # Add a legend for node colors and sizes
    legend_elements = []
    
    # Add color legend
    for node_type, color in NODE_COLORS.items():
        if node_type in nodes:
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor=color, 
                             alpha=0.8, label=f"{node_type}")
            )
    
    # Add size legend for sensitivity if available
    if sensitivity_data:
        for size, label in [(1000, 'Low Impact'), (3000, 'Medium Impact'), (5000, 'High Impact')]:
            legend_elements.append(
                plt.scatter([], [], s=size, color='gray', alpha=0.5, label=label)
            )
    
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{station_code}_bn_structure.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.tight_layout()
    else:
        plt.close(fig)
    
    return fig 


def plot_cpt_heatmap(
    station_data: Dict,
    node: str = 'Compound_Flood_Risk',
    ax: Optional[plt.Axes] = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize a Conditional Probability Table (CPT) as a heatmap.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    node : str, optional
        Node name to visualize CPT for, default is Compound_Flood_Risk
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if CPT statistics are available
    if ('tier3_analysis' not in station_data or 
        'bayesian_network' not in station_data['tier3_analysis'] or
        'cpt_statistics' not in station_data['tier3_analysis']['bayesian_network']):
        logger.warning("No CPT statistics available")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
            ax.text(0.5, 0.5, "No CPT statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No CPT statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract CPT statistics
    cpt_stats = station_data['tier3_analysis']['bayesian_network']['cpt_statistics']
    
    # Check if the requested node is available
    if node not in cpt_stats:
        logger.warning(f"No CPT statistics available for node {node}")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
            ax.text(0.5, 0.5, f"No CPT statistics available for node {node}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, f"No CPT statistics available for node {node}", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract most likely states
    most_likely_states = cpt_stats[node].get('most_likely_states', {})
    
    # If no parent combinations exist (marginal distribution)
    if 'marginal' in most_likely_states:
        # Create a simple bar chart for the marginal distribution
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        else:
            fig = ax.figure
        
        # Create a simple dataframe for the marginal distribution
        marginal_data = []
        
        # Extract states from node_states if available
        bn_structure = station_data['tier3_analysis']['bayesian_network'].get('structure', {})
        node_states = bn_structure.get('node_states', {}).get(node, [])
        
        if not node_states:
            # Fallback if node_states not available
            if node == 'Compound_Flood_Risk':
                node_states = ['Low', 'Moderate', 'High']
            elif node in ['Sea_Level_State', 'Precipitation_State']:
                node_states = ['Below_90th', '90th_99th', 'Above_99th']
            else:
                # Create generic states
                node_states = [f'State_{i}' for i in range(3)]
        
        # Create probability bars for each state
        probs = []
        for state in node_states:
            # Check if this state is the most likely one
            if most_likely_states['marginal']['state'] == state:
                probs.append(most_likely_states['marginal']['probability'])
            else:
                # Assign a low probability by default
                probs.append(0.1)
        
        # Normalize to ensure sum = 1.0
        if sum(probs) > 0:
            probs = [p / sum(probs) for p in probs]
        
        # Plot bar chart
        sns.barplot(x=node_states, y=probs, ax=ax, palette=RISK_CMAP)
        
        # Add labels
        ax.set_xlabel(f'{node} States')
        ax.set_ylabel('Probability')
        
        # Add a title
        station_code = station_data.get('station', '')
        ax.set_title(f"{node} Marginal Distribution - {station_code}")
    
    else:
        # For conditional distributions, create a heatmap
        
        # Create a dataframe from the most likely states info
        heatmap_data = []
        
        # Parse parent and state combinations
        for combo_str, combo_info in most_likely_states.items():
            # Skip if the combo_str is 'error' or similar non-combo string
            if not '_' in combo_str:
                continue
                
            # Parse the combo string (format: "ParentName1=State1_ParentName2=State2_...")
            parent_state_pairs = combo_str.split('_')
            parent_states = {}
            for pair in parent_state_pairs:
                parts = pair.split('=')
                if len(parts) == 2:
                    parent_states[parts[0]] = parts[1]
            
            # Add states and probability
            state = combo_info['state']
            prob = combo_info['probability']
            
            # Extract values to create a row
            row = parent_states.copy()
            row['state'] = state
            row['probability'] = prob
            heatmap_data.append(row)
        
        # Convert to DataFrame
        if not heatmap_data:
            # If no valid data found, show a message
            if ax is None:
                fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
            else:
                fig = ax.figure
            
            ax.text(0.5, 0.5, "No valid CPT combinations found for visualization", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        df = pd.DataFrame(heatmap_data)
        
        # Check if we need to create a pivot table (2 or more parents)
        if len(df.columns) > 3:  # state, probability, and at least 2 parents
            # Identify the columns that represent parents
            parent_cols = [col for col in df.columns if col not in ['state', 'probability']]
            
            # Select two parents to use for the pivot
            # Prioritize Sea_Level_State and Precipitation_State if available
            if 'Sea_Level_State' in parent_cols and 'Precipitation_State' in parent_cols:
                pivot_cols = ['Sea_Level_State', 'Precipitation_State']
            else:
                # Otherwise use the first two parents
                pivot_cols = parent_cols[:2]
            
            # Create a pivot table
            pivot_df = df.pivot_table(
                values='probability', 
                index=pivot_cols[0], 
                columns=pivot_cols[1],
                aggfunc='mean'  # Use mean in case there are multiple entries
            )
            
            # Create figure
            if ax is None:
                fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
            else:
                fig = ax.figure
            
            # Create heatmap
            sns.heatmap(
                pivot_df, 
                annot=True, 
                cmap=CPR_CMAP, 
                fmt='.2f', 
                ax=ax
            )
            
            # Add a title
            station_code = station_data.get('station', '')
            pivot_title = f"{node} CPT - {pivot_cols[0]} vs {pivot_cols[1]} - {station_code}"
            ax.set_title(pivot_title)
            
        else:
            # For a single parent, create a simpler visualization
            # Find the parent column
            parent_col = [col for col in df.columns if col not in ['state', 'probability']]
            if parent_col:
                parent_col = parent_col[0]
                
                # Create figure
                if ax is None:
                    fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
                else:
                    fig = ax.figure
                
                # Create a grouped bar chart
                sns.barplot(
                    x=parent_col, 
                    y='probability', 
                    hue='state', 
                    data=df, 
                    ax=ax,
                    palette=RISK_CMAP
                )
                
                # Add a title
                station_code = station_data.get('station', '')
                ax.set_title(f"{node} CPT given {parent_col} - {station_code}")
                
                # Improve layout
                ax.set_xlabel(parent_col)
                ax.set_ylabel('Probability')
                ax.legend(title=f"{node} State")
                
            else:
                # Fallback if structure is unexpected
                if ax is None:
                    fig, ax = plt.subplots(figsize=FIG_SIZES['square'])
                else:
                    fig = ax.figure
                
                ax.text(0.5, 0.5, "Unexpected CPT format for visualization", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        station_code = station_data.get('station', '')
        filename = os.path.join(output_dir, f"{station_code}_{node}_cpt.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.tight_layout()
    else:
        plt.close(fig)
    
    return fig


def plot_sensitivity_analysis(
    station_data: Dict,
    ax: Optional[plt.Axes] = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the sensitivity analysis results.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if sensitivity data is available
    if ('tier3_analysis' not in station_data or 
        'bayesian_network' not in station_data['tier3_analysis'] or
        'sensitivity' not in station_data['tier3_analysis']['bayesian_network'] or
        not station_data['tier3_analysis']['bayesian_network']['sensitivity']):
        
        logger.warning("No sensitivity analysis data available")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
            ax.text(0.5, 0.5, "No sensitivity analysis data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No sensitivity analysis data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract sensitivity data
    sensitivity_data = station_data['tier3_analysis']['bayesian_network']['sensitivity']
    
    # Create a dataframe from the sensitivity data
    sensitivity_df = []
    
    for node, metrics in sensitivity_data.items():
        if isinstance(metrics, dict) and 'average_impact' in metrics:
            row = {
                'node': node,
                'average_impact': metrics['average_impact'],
                'max_impact': metrics.get('max_impact', metrics['average_impact'])
            }
            sensitivity_df.append(row)
    
    # Convert to DataFrame
    if not sensitivity_df:
        logger.warning("No valid sensitivity data available for visualization")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
            ax.text(0.5, 0.5, "No valid sensitivity data available for visualization", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No valid sensitivity data available for visualization", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    sensitivity_df = pd.DataFrame(sensitivity_df)
    
    # Sort by average impact
    sensitivity_df = sensitivity_df.sort_values('average_impact', ascending=False)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
    else:
        fig = ax.figure
    
    # Create a horizontal bar chart
    node_colors = [NODE_COLORS.get(node, '#aaaaaa') for node in sensitivity_df['node']]
    bars = ax.barh(sensitivity_df['node'], sensitivity_df['average_impact'], color=node_colors, alpha=0.8)
    
    # Add max impact markers
    for i, (_, row) in enumerate(sensitivity_df.iterrows()):
        ax.plot([row['max_impact']], [i], 'ro', alpha=0.7)
    
    # Add a title
    station_code = station_data.get('station', '')
    ax.set_title(f"Sensitivity Analysis - {station_code}")
    
    # Add labels
    ax.set_xlabel('Sensitivity (Average Impact on Risk)')
    ax.set_ylabel('Node')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", va='center')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a note about max impact markers
    ax.text(0.98, 0.02, "Red markers show max impact", 
            transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{station_code}_sensitivity.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.tight_layout()
    else:
        plt.close(fig)
    
    return fig 


def plot_risk_distribution(
    station_data: Dict,
    ax: Optional[plt.Axes] = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the distribution of risk classes.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if class distribution data is available
    if ('tier3_analysis' not in station_data or 
        'bayesian_network' not in station_data['tier3_analysis'] or
        'validation' not in station_data['tier3_analysis']['bayesian_network'] or
        'class_distribution' not in station_data['tier3_analysis']['bayesian_network']['validation']):
        
        logger.warning("No risk class distribution data available")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
            ax.text(0.5, 0.5, "No risk class distribution data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No risk class distribution data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract class distribution data
    class_dist = station_data['tier3_analysis']['bayesian_network']['validation']['class_distribution']
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'class': ['Low', 'Moderate', 'High'],
        'count': [class_dist.get('low', 0), class_dist.get('moderate', 0), class_dist.get('high', 0)]
    })
    
    # Calculate percentages
    df['percentage'] = 100 * df['count'] / df['count'].sum()
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
    else:
        fig = ax.figure
    
    # Define colors for each risk level
    colors = ['#ffffcc', '#a1dab4', '#253494']
    
    # Create bar chart using matplotlib instead of seaborn
    bars = ax.bar(df['class'], df['count'], color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{df["percentage"].iloc[i]:.1f}%',
                ha='center', fontsize=10)
    
    # Add title
    station_code = station_data.get('station', '')
    ax.set_title(f"Risk Class Distribution - {station_code}")
    
    # Improve layout
    ax.set_xlabel('Risk Class')
    ax.set_ylabel('Count')
    
    # Set y-axis to log scale if there's high class imbalance
    if max(df['count']) / min(df['count'] if min(df['count']) > 0 else 1) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Count (log scale)')
    
    # Add total count as text
    total = df['count'].sum()
    ax.text(0.98, 0.95, f"Total: {total:,}", transform=ax.transAxes, ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{station_code}_risk_distribution.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.tight_layout()
    else:
        plt.close(fig)
    
    return fig


def plot_validation_metrics(
    station_data: Dict,
    ax: Optional[plt.Axes] = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the validation metrics for the Bayesian Network.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if validation metrics are available
    if ('tier3_analysis' not in station_data or 
        'bayesian_network' not in station_data['tier3_analysis'] or
        'validation' not in station_data['tier3_analysis']['bayesian_network']):
        
        logger.warning("No validation metrics available")
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
            ax.text(0.5, 0.5, "No validation metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        else:
            ax.text(0.5, 0.5, "No validation metrics available", 
                   ha='center', va='center', transform=ax.transAxes)
            return ax.figure
    
    # Extract validation metrics
    validation = station_data['tier3_analysis']['bayesian_network']['validation']
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
    else:
        fig = ax.figure
    
    # Create metrics dataframe
    metrics = [
        {'name': 'Log Likelihood', 'value': validation.get('log_likelihood', 0)},
        {'name': 'Accuracy', 'value': validation.get('accuracy', 0)},
        {'name': 'Brier Score', 'value': validation.get('brier_score', 0)}
    ]
    
    df = pd.DataFrame(metrics)
    
    # Create horizontal bar chart
    bars = ax.barh(df['name'], df['value'], color=['#1f77b4', '#2ca02c', '#d62728'])
    
    # Add a title
    station_code = station_data.get('station', '')
    ax.set_title(f"Validation Metrics - {station_code}")
    
    # Add value annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01 if width > 0 else width - 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}", 
                va='center', 
                ha='left' if width > 0 else 'right')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{station_code}_validation_metrics.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.tight_layout()
    else:
        plt.close(fig)
    
    return fig


def plot_station_tier3_summary(
    station_data: Dict,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary visualization for a station.
    
    Parameters
    ----------
    station_data : Dict
        Tier 3 results for a single station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Set publication style
    set_publication_style()
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES['full'])
    
    # Extract station code
    station_code = station_data.get('station', '')
    fig.suptitle(f"Tier 3 Bayesian Network Summary - Station {station_code}", fontsize=16)
    
    # Plot network structure
    plot_bayesian_network(station_data, ax=axes[0, 0], show=False)
    
    # Plot sensitivity analysis
    plot_sensitivity_analysis(station_data, ax=axes[0, 1], show=False)
    
    # Plot risk distribution
    plot_risk_distribution(station_data, ax=axes[1, 0], show=False)
    
    # Plot CPT heatmap for the risk node
    plot_cpt_heatmap(station_data, node='Compound_Flood_Risk', ax=axes[1, 1], show=False)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{station_code}_tier3_summary.png")
        save_figure(fig, filename)
    
    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_tier3_summary_report(
    tier3_data: Dict,
    output_dir: str,
    station_codes: List[str] = None,
    show: bool = False
) -> None:
    """
    Create summary visualizations for multiple stations.
    
    Parameters
    ----------
    tier3_data : Dict
        Dictionary of Tier 3 results, keyed by station code
    output_dir : str
        Directory to save figures
    station_codes : List[str], optional
        List of station codes to process. If None, process all stations.
    show : bool, optional
        Whether to display the figures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    set_publication_style()
    
    # Get station codes to process
    if station_codes is None:
        station_codes = list(tier3_data.keys())
    
    # Process each station
    for i, station_code in enumerate(station_codes):
        logger.info(f"Processing station {station_code} ({i+1}/{len(station_codes)})")
        
        # Check if we have data for this station
        if station_code not in tier3_data:
            logger.warning(f"No Tier 3 data available for station {station_code}")
            continue
        
        # Get station data
        station_data = tier3_data[station_code]
        
        # Create a directory for this station
        station_dir = os.path.join(output_dir, station_code)
        os.makedirs(station_dir, exist_ok=True)
        
        try:
            # Create comprehensive summary plot
            plot_station_tier3_summary(station_data, output_dir=station_dir, show=False)
            
            # Create individual plots for different components
            plot_bayesian_network(station_data, output_dir=station_dir, show=False)
            plot_sensitivity_analysis(station_data, output_dir=station_dir, show=False)
            plot_risk_distribution(station_data, output_dir=station_dir, show=False)
            plot_validation_metrics(station_data, output_dir=station_dir, show=False)
            
            # Plot CPTs for various nodes
            bn_structure = station_data.get('tier3_analysis', {}).get(
                'bayesian_network', {}).get('structure', {})
            nodes = bn_structure.get('nodes', [])
            
            for node in nodes:
                plot_cpt_heatmap(station_data, node=node, output_dir=station_dir, show=False)
                
        except Exception as e:
            logger.error(f"Error creating visualizations for station {station_code}: {e}")
            continue
    
    logger.info(f"Completed creating Tier 3 summary reports for {len(station_codes)} stations")


if __name__ == "__main__":
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Create Tier 3 Bayesian Network visualizations')
    parser.add_argument('--tier3-dir', default='outputs/tier3', 
                       help='Directory containing Tier 3 outputs')
    parser.add_argument('--output-dir', default='outputs/plots/tier3', 
                       help='Directory to save visualizations')
    parser.add_argument('--station-code', help='Specific station code to process')
    parser.add_argument('--show', action='store_true', help='Display plots')
    
    args = parser.parse_args()
    
    # Load Tier 3 results
    if args.station_code:
        tier3_data = load_tier3_results(args.tier3_dir, args.station_code)
        if args.station_code in tier3_data:
            # Create summary for a single station
            station_data = tier3_data[args.station_code]
            plot_station_tier3_summary(station_data, output_dir=args.output_dir, show=args.show)
            logger.info(f"Created summary for station {args.station_code}")
        else:
            logger.error(f"No Tier 3 data found for station {args.station_code}")
    else:
        # Load all stations
        tier3_data = load_tier3_results(args.tier3_dir)
        if tier3_data:
            # Create summaries for all stations
            create_tier3_summary_report(tier3_data, args.output_dir, show=args.show)
            logger.info(f"Created summaries for {len(tier3_data)} stations")
        else:
            logger.error(f"No Tier 3 data found in {args.tier3_dir}")