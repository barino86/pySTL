from typing import Union

import pandas as pd
from anomaly_detection.main.utils import get_gran_and_period


class STLDecomp(object):

    def __init__(self, stl_data: pd.DataFrame, x_axis: Union[pd.Series, None] = None):
        reqd_cols = ['value', 'seasonal', 'trend', 'remainder']

        if not isinstance(stl_data, pd.DataFrame) or stl_data.empty:
            raise ValueError("'stl_data' argument must be a non-empty dataframe.")

        if stl_data.shape[1] != 4 or any([x not in reqd_cols for x in stl_data.columns]):
            raise ValueError(f"Dataframe must have all of the required columns: {reqd_cols}")

        self.data: pd.DataFrame = stl_data
        if not isinstance(x_axis, pd.Series) or x_axis.empty:
            self.__x_axis = list(range(self.data.shape[0]))
        else:
            self.__x_axis = x_axis

    @property
    def x_axis(self):
        return self.__x_axis

    @x_axis.setter
    def x_axis(self, value: pd.Series):
        if len(list(value)) != self.data.shape[0]:
            raise ValueError(f"Length of the x_axis must be the same as the length of the data: {self.data.shape[0]}")
        self.__x_axis = value

    def _build_plot(self):
        import matplotlib.pyplot as plt
        from pandas._libs.tslibs.timestamps import Timestamp
        plt.style.use('ggplot')
        fig, (raw, seas, trend, rem) = plt.subplots(nrows=4, ncols=1, figsize=(7, 8), sharex=True)

        # Build Raw Plot
        raw.plot(self.x_axis, self.data['value'], color="#00AED9", linewidth=1)

        # Only add labels to points where the seasonal is the maximum.
        max_seas = max(self.data['seasonal'])
        iter_map = filter(lambda x: x[2] == max_seas, zip(self.x_axis, self.data['value'], self.data['seasonal']))
        for i, row in enumerate(iter_map):
            raw.scatter(row[0], row[1], marker=".", color="#FEB948", s=30)
            if i % 2 == 0:
                raw.annotate(f"{row[1]:.0f}",
                             (row[0], row[1]),
                             textcoords="offset points",
                             xytext=(0, 5),
                             ha='center',
                             size=8,
                             color="#184C6D")
        raw.set_ylabel("raw")

        # Build Seasonal Plot
        seas.plot(self.x_axis, self.data['seasonal'], color="#184C6D")
        seas.set_ylabel("seasonal")

        # Build Trend Plot
        trend.plot(self.x_axis, self.data['trend'], color="#00AED9")
        trend.set_ylabel("trend")

        # Build remainder Plot with x_axis formatted correctly.
        x_axis_is_ts = False
        gran, period = None, 1
        if isinstance(self.x_axis[0], Timestamp):
            x_axis_is_ts = True
            gran, period = get_gran_and_period(self.x_axis)
            if gran == 'week':
                period = 1 / 7

        bar_width = 1 / period

        x_axis = self.x_axis
        if x_axis_is_ts:
            x_axis = x_axis.values
        rem.bar(x_axis, self.data['remainder'], width=bar_width)
        rem.set_ylabel("remainder")

        if x_axis_is_ts:
            import matplotlib.dates as mdates
            fmt = mdates.DateFormatter('%Y-%m-%d')
            rem.xaxis.set_major_formatter(fmt)
            fig.autofmt_xdate()

        return fig

    def display_plot(self) -> None:
        fig = self._build_plot()
        fig.show()

    def save_plot(self, image_path: str = None) -> None:
        fig = self._build_plot()
        fig.savefig(image_path)
        print(f"Plot Saved to '{image_path}'")


class AnomDetect(object):

    def __init__(self, orig: pd.DataFrame, labeled: pd.DataFrame, anoms: pd.DataFrame, stl: STLDecomp):
        self.original: pd.DataFrame = orig
        self.labeled_data: pd.DataFrame = labeled
        self.anomalies: pd.DataFrame = anoms
        self.anom_percentage = anoms.shape[0] / labeled.shape[0]
        self.stl: STLDecomp = stl

    def _build_plot(self):
        import matplotlib.pyplot as plt
        from pandas._libs.tslibs.timestamps import Timestamp
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(self.labeled_data['timestamp'], self.labeled_data['value'], color="#00AED9", linewidth=1, zorder=1)
        for ix, row in enumerate(self.labeled_data.itertuples(index=False)):

            # Anomaly points
            if row.label != 'Normal':
                ax.scatter(row.timestamp, row.value,
                           marker=".", color=("#6A9C00" if row.label == 'High' else "#D45756"),
                           edgecolors="#184C6D", s=100, zorder=2, alpha=0.8)
                ax.annotate(f"{row.value:.0f}", (row.timestamp, row.value),
                            textcoords="offset points", xytext=(-15, 0), ha='center', size=8, alpha=.7)

            # Every 10 points if the point is not an anomaly
            elif ix % 10 == 0:
                ax.scatter(row.timestamp, row.value, marker=".", color="#FEB948", s=80, zorder=2)
                ax.annotate(f"{row.value:.0f}", (row.timestamp, row.value),
                            textcoords="offset points", xytext=(0, 5), ha='center', size=6, color="#184C6D")

        # format x axis if x axis is timestamp
        if isinstance(self.labeled_data['timestamp'][0], Timestamp):
            import matplotlib.dates as mdates
            fmt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(fmt)
            fig.autofmt_xdate()

        return fig

    def display_plot(self) -> None:
        fig = self._build_plot()
        fig.show()

    def save_plot(self, image_path: str = None) -> None:
        fig = self._build_plot()
        fig.savefig(image_path)
        print(f"Plot Saved to '{image_path}'")
