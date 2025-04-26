import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as st

# Initial details
save_path = 'C:\\Users\\michael.moore\\OneDrive - RSSB\\Python scripts'
date = datetime.today().strftime('%Y-%m-%d')
name = '2024-DWG-Analysis' + '-' + date

# Module 1.1 - parameters to update for ISLG
year = '2023/24'
startperiod = 202101
endperiod = 202313
startperiod = int(startperiod)  # Convert to integer if not already
endperiod = int(endperiod)
N = 39
fyearpoints = [6.5, 19.5, 31.5]
bigticks = [-0.5, 12.5, 25.5, N - 0.5]
bigticks2 = [-0.5, 12.5, 25.5, N - 0.5]
xlimitend = N - 0.5
ylimitend = 50

# Module 3 - Load CSV instead of SQL
csv_path = 'C:\\Users\\mmoor\\OneDrive\\SQL\\roi_depot_incidents.csv'

# Read CSV into DataFrame
smis_df1 = pd.read_csv(csv_path)

# Group data for counts
# Count total events by period
smis_counts1 = smis_df1.groupby(['period'])['smis_reference'].count().fillna(0)

# Create a DataFrame for results
Nyears = 3  # Number of years in graph
ymax = int(year[:4])
ymin = int(ymax) - Nyears + 1
for i in range(Nyears + 1):
    p1 = np.arange(1, 14)
    y1 = int(ymax) - Nyears + i
    y2 = y1 + 1
    data = pd.DataFrame({'Periods': p1.astype(str)})
    data['Index'] = str(y1) + data['Periods'].str.zfill(2)
    data['Index'] = data['Index'].astype(int)
    data['Periods'] = 'P' + data['Periods']
    data['Fiscal year'] = str(y1) + '/' + str(y2)
    if i == 0:
        data1 = data
    else:
        data1 = pd.concat([data1, data], sort=True)

data1 = data1.set_index('Index')

# Join with counts
smis_data2 = data1.join(smis_counts1.rename("TotalROIIncidents"))
smis_data2 = smis_data2.fillna(0)

# Calculate annual moving average
smis_data2['ROI Events AMA'] = smis_data2['TotalROIIncidents'].rolling(window=13).mean()

# Filter data by the defined period
smis_data2 = smis_data2[smis_data2.index >= startperiod]
smis_data2 = smis_data2[smis_data2.index <= endperiod]

###################################################
# Module 6.1 - calculating upper and lower bounds
###################################################
conf_intervals_cut = smis_counts1[(smis_counts1.index >= 202101) & (smis_counts1.index <= 202213)]
conf_intervals_mean = np.mean(conf_intervals_cut)

bounds = st.norm.interval(confidence=0.95, loc=conf_intervals_mean, scale=st.sem(conf_intervals_cut))
lowerbound = bounds[0]
upperbound = bounds[1]

###################################################
# Module 7 - draw chart
###################################################

# Set font parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Calibri'
plt.rcParams['font.size'] = 15.0

# Parameters for graph
fig = plt.figure(figsize=(13, 6))
width = 0.5
ind = np.arange(0, N, 1)
bordercolour = '0.6'
graphtitle = 'Trend in Railway Operating Incidents (ROI)'

# x tick mark locations for all graphs
ticks1 = np.arange(0, N, 1)  # Set tick intervals
ticklabels1 = smis_data2['Periods'].iloc[ticks1].tolist()  # Labels at correct intervals
fyears = '\n' + smis_data2['Fiscal year'].dropna().unique()
ticklabels1 = ticklabels1 + fyears.tolist()
ticks1 = ticks1.tolist() + fyearpoints

# All ROI graph
ax1 = fig.add_subplot(1, 1, 1)
ax1.set(ylabel='Number of events')
ax1.set_xlim(-0.5, xlimitend)
ax1.set_ylim(0, ylimitend)
ax1.set_xticks(ticks1, minor=True)
ax1.set_xticklabels(ticklabels1, minor=True)
ax1.set_xticks(bigticks)
ax1.set_xticklabels([])
ax1.tick_params(axis='x', which='minor', length=0)
ax1.tick_params(axis='x', which='major', length=30, color=bordercolour)

# Gridlines
ax1.yaxis.grid(which="major", color=bordercolour, linestyle='-', linewidth=1, zorder=0)

# Plot bar graph
ax1.bar(ind, smis_data2['TotalROIIncidents'], width, color=(33/255, 33/255, 79/255), zorder=3, label='Total ROI Incidents')
ax1.plot(ind, smis_data2['ROI Events AMA'], color='green', label='Total annual moving average (AMA)')

# Confidence intervals
ax1.plot([-0.5, xlimitend], [upperbound, upperbound], ls='--', color='black', label='Upper bound (95% confidence)')
ax1.plot([-0.5, xlimitend], [lowerbound, lowerbound], ls='--', color='black', label='Lower bound (95% confidence)')

# Final formatting
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_color(bordercolour)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3)
ax1.set_facecolor('whitesmoke')
fig.patch.set_facecolor('whitesmoke')

plt.title(graphtitle)
plt.show()
