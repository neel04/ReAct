from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import time

def plot_freq(errors: List[int], epoch: int):
    """Plot the frequency of errors generated during training."""

    error_counter = Counter(errors)
    # Extract unique errors and their corresponding frequencies
    unique_errors = list(error_counter.keys())
    frequencies = list(error_counter.values())

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define a custom color for the bars (lighter green with transparency)
    bar_color = (0.4, 0.8, 0.4, 0.6)

    # Create a bar plot with the custom color
    bars = ax.bar(unique_errors, frequencies, color=bar_color)

    # Add data labels to the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', color='black', fontsize=10)

    # Customize plot appearance
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of Errors')
    ax.set_xticks(unique_errors)
    ax.set_facecolor('lightgray')

    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add a background color to the plot area
    fig.patch.set_facecolor('white')

    # Show the plot
    plt.tight_layout()
    #plt.savefig(f'/fsx/awesome/DPT/outputs/errors_{epoch}.png')
    plt.close()

    return error_counter, fig