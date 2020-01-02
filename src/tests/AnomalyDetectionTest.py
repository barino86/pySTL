import unittest
import pandas as pd
from main.detect_anoms import detect_anoms, build_plot


class AnomalyDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = '/Users/brandonarino/Desktop/test_hourly.csv'
        self.data = pd.read_csv(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        self.period = 24
        self.data_seasonal = True
        self.one_tail = True
        self.upper_tail = True

    def test_detect_anoms(self):
        data_decomp, anoms, stl = detect_anoms(
            self.data, period=self.period, data_seasonal=self.data_seasonal,
            one_tail=self.one_tail, upper_tail=self.upper_tail
        )
        print(data_decomp)
        print(anoms)
        print(stl)
        fig = build_plot(data_decomp)
        fig.show()


if __name__ == '__main__':
    unittest.main()
