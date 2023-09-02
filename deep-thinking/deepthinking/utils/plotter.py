import matplotlib.pyplot as plt
import io
import time

from collections import Counter
from typing import List, Tuple
from PIL import Image

def plot_freq(errors: List[int], epoch: int) -> Tuple[Counter, Image.Image]:
    """Plot the frequency of errors generated during training and return a PIL Image."""
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

    # Save the figure to a BytesIO object
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Give some time for the file to be written
    time.sleep(1)  # Adjust the sleep duration as needed

    # Convert the BytesIO object to a PIL Image
    pil_image = Image.open(image_stream)

    return error_counter, pil_image