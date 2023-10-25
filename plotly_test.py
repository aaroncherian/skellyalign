import plotly.graph_objects as go

# Simulated RMSE values for X, Y, Z dimensions for a set of joints
rmse_x = [1, 2, 3, 4]
rmse_y = [2, 3, 4, 1]
rmse_z = [3, 4, 1, 2]

# Simulated positions for joints
joints_x = [1, 2, 3, 4]
joints_y = [2, 3, 4, 5]

# Colors representing RMSE values for each dimension
colors_x = ['red'] * len(rmse_x)
colors_y = ['green'] * len(rmse_y)
colors_z = ['blue'] * len(rmse_z)

fig = go.Figure()

# Add scatter plots for each dimension
fig.add_trace(go.Scatter(x=joints_x, y=joints_y, mode='markers',
                         marker=dict(size=rmse_x, color=colors_x), name='X'))
fig.add_trace(go.Scatter(x=joints_x, y=joints_y, mode='markers',
                         marker=dict(size=rmse_y, color=colors_y), name='Y'))
fig.add_trace(go.Scatter(x=joints_x, y=joints_y, mode='markers',
                         marker=dict(size=rmse_z, color=colors_z), name='Z'))

# Add axis labels and title
fig.update_layout(
    title="Joint RMSEs by Dimension",
    xaxis_title="Joint X Coordinate",
    yaxis_title="Joint Y Coordinate",
    xaxis=dict(
    tickfont=dict(
        size=14
        )
    )
)

fig.show()