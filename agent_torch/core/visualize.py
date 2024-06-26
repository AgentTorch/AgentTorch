import pandas as pd
import random
import folium


class Visualizer:
    def __init__(self, state_trace_df):
        self.state_trace_df = state_trace_df

    def plot(self, var_name):
        data = {
            "area": [100100, 100200, 100300, 100400, 100500],
            "latitude": [-34.505453, -34.916277, -35.218501, -34.995278, -35.123147],
            "longitude": [172.775550, 173.137443, 174.158249, 173.378738, 173.218604],
            "2023-01-01": [5.2, 6.7, 4.1, 8.3, 3.9],
            "2023-02-01": [4.9, 6.3, 3.8, 7.8, 4.2],
            "2023-03-01": [5.5, 7.1, 4.4, 8.6, 4.1],
        }

        df = pd.DataFrame(data)
        if var_name == "unemployment_rate":
            map = folium.Map(location=[-40, 172], zoom_start=5)
            # Add a marker for each point in the DataFrame for each month
            for idx, row in df.iterrows():
                color = self.get_color(row["2023-01-01"])
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=20,
                    color=color,
                    fill=True,
                    fill_color=color,
                ).add_to(map)

            return map

        elif var_name == "will_to_work":
            map = folium.Map(location=[-40, 172], zoom_start=5)
            # Add a marker for each point in the DataFrame for each month
            for idx, row in df.iterrows():
                color = self.get_color(row["2023-02-01"])
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=20,
                    color=color,
                    fill=True,
                    fill_color=color,
                ).add_to(map)

            return map

    # Function to map values to colors
    def get_color(self, value):
        if value > 8:
            return "darkred"
        elif value > 6:
            return "red"
        elif value > 5:
            return "orange"
        elif value > 4:
            return "yellow"
        else:
            return "green"
