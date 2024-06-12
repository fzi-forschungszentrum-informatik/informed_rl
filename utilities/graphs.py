import json
import matplotlib
import pandas as pd
import numpy as np
#matplotlib.use("pgf")
import matplotlib.pyplot as plt

""" matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8
}) """

def filter_data(task, threshold=40000):
    filtered_x = [x for x in task["x"] if x <= threshold]
    filtered_y = [task["y"][i] for i, x in enumerate(task["x"]) if x <= threshold]
    return {"name": task["name"], "x": filtered_x, "y": filtered_y, "type": task["type"], "task": task["task"]}

# Define a function to compute exponential moving average
def exponential_moving_average(data, span):
    return pd.DataFrame(data).ewm(span=span, adjust=False).mean().values.flatten()

def moving_std_pos_neg(data, avg_data, min_periods, window_size):
    diff = np.array(data) - np.array(avg_data)
    diff_pos = np.where(diff >= 0, diff, 0)
    diff_neg = np.where(diff < 0, -diff, 0)
    expanding_std_pos = pd.DataFrame(diff_pos).expanding(min_periods=min_periods).std()
    expanding_std_neg = pd.DataFrame(diff_neg).expanding(min_periods=min_periods).std()
    rolling_std_pos = pd.DataFrame(diff_pos).rolling(window_size).std()
    rolling_std_neg = pd.DataFrame(diff_neg).rolling(window_size).std()
    std_pos = pd.concat([expanding_std_pos[:window_size], rolling_std_pos[window_size:]]).fillna(0).values.flatten()
    std_neg = pd.concat([expanding_std_neg[:window_size], rolling_std_neg[window_size:]]).fillna(0).values.flatten()
    return std_pos, std_neg

def running_min(data, min_periods, window_size):
    expanding_min = pd.DataFrame(data).expanding(min_periods=min_periods).min()
    rolling_min = pd.DataFrame(data).rolling(window_size).min()
    return pd.concat([expanding_min[:window_size], rolling_min[window_size:]]).fillna(0).values.flatten()

def running_max(data, min_periods, window_size):
    expanding_max = pd.DataFrame(data).expanding(min_periods=min_periods).max()
    rolling_max = pd.DataFrame(data).rolling(window_size).max()
    return pd.concat([expanding_max[:window_size], rolling_max[window_size:]]).fillna(0).values.flatten()

def running_quantile(data, min_periods, window_size, quantile):
    expanding_quantile = pd.DataFrame(data).expanding(min_periods=min_periods).quantile(quantile)
    rolling_quantile = pd.DataFrame(data).rolling(window_size).quantile(quantile)
    return pd.concat([expanding_quantile[:window_size], rolling_quantile[window_size:]]).fillna(0).values.flatten()

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def plot_data(ax, task_data, linestyle='-o'):
    for task_name, task_info in task_data.items():
        #ax.plot(task_info["x"], task_info["y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.1',markersize=0,alpha=0.3)
        ax.plot(task_info["x"], task_info["avg_y"], linestyle, label=task_name, color=task_info["color"], linewidth = '2',markersize=0)
        ax.fill_between(task_info["x"][::10], task_info["avg_y"][::10] - task_info["std_y_neg"][::10], task_info["avg_y"][::10] + task_info["std_y_pos"][::10], color=task_info["color"], alpha=0.2)  # Plot standard deviation
        #ax.plot(task_info["x"], task_info["min_y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.5',markersize=0)  # Plot running minimum
        #ax.plot(task_info["x"], task_info["max_y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.5',markersize=0)  # Plot running maximum
        ax.fill_between(task_info["x"][::10], task_info["5th_percentile"][::10], task_info["95th_percentile"][::10], color=task_info["color"], alpha=0.1)  # Plot 5th and 95th percentiles
        
# Load data from the files
file_paths = [
    'train_arrived_s _ train_arrived_s.json',
    'train_finished_score _ train_finished_score.json'
]

data_1 = [filter_data(task) for task in load_data(file_paths[0])]
data_2 = [filter_data(task) for task in load_data(file_paths[1])]


# Create a dictionary to store data for each task
task_data_1 = {}
task_data_2 = {}


# Define a list of colors to use for each task
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
task_names = ["Dreamer + Trajectory + Rulebook", "Dreamer + Trajectory", "Dreamer + Rulebook", "Dreamer", "Rainbow + Rulebook", "Rainbow"]

# Process filtered data
for i, (filtered_data, task_data) in enumerate(zip([data_1, data_2], [task_data_1, task_data_2])):
    for j, item in enumerate(filtered_data):
        #task_name = item["task"]
        task_name = task_names[j]
        color = colors[j % len(colors)]
        if task_name not in task_data:
            task_data[task_name] = {"x": [], "y": [], "avg_y": [], "std_y_neg": [], "std_y_pos": [], "min_y": [], "max_y": [], "5th_percentile": [], "95th_percentile": [], "name": item["name"], "color": color}

        task_data[task_name]["x"].extend(item["x"])
        task_data[task_name]["y"].extend(item["y"])
        task_data[task_name]["avg_y"] = exponential_moving_average(item["y"], span=100)  # Compute running average
        task_data[task_name]["std_y_pos"], task_data[task_name]["std_y_neg"] = moving_std_pos_neg(item["y"], task_data[task_name]["avg_y"], min_periods=2, window_size=100)  # Compute moving standard deviation
        task_data[task_name]["min_y"] = running_min(item["y"], min_periods=1, window_size=100)  # Compute running minimum
        task_data[task_name]["max_y"] = running_max(item["y"], min_periods=1, window_size=100)  # Compute running maximum
        task_data[task_name]["5th_percentile"] = running_quantile(item["y"], min_periods=1, window_size=100, quantile=0.05)  # Compute running 5th percentile
        task_data[task_name]["95th_percentile"] = running_quantile(item["y"], min_periods=1, window_size=100, quantile=0.95)  # Compute running 95th percentile


# Plot the data for each task in subplots
subplot_width = 3.4375  # 6.875 inches divided by 4 subplots

fig, axs = plt.subplots(1, 2, figsize=(6.875, subplot_width), gridspec_kw={'width_ratios': [subplot_width] * 2})
#fig.set_size_inches(w=6.875, h=1.5)

plot_data(axs[0], task_data_1)
plot_data(axs[1], task_data_2)


# Set titles and labels
axs[0].set_title(f"Arrived Distance")
axs[1].set_title(f"Finished Score")
axs[0].set_ylabel(f"Distance [m] $\\uparrow$")
axs[1].set_ylabel(f"Score $\\uparrow$")

for ax in axs:
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Only show legend in the top-left subplot
axs[0].legend(fontsize=5,fancybox=True,loc='lower right',bbox_to_anchor=(0.5,0.02,0.46,1),mode="expand")

# Save the plot
fig.tight_layout()
#plt.savefig('train.pdf')
plt.savefig("train.svg", format="svg")
#plt.savefig('train.pgf')