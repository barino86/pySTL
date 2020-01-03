import unittest
import pandas as pd
from anomaly_detection.main.AnomalyDetection import AnomalyDetection


class AnomalyDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_path = '/Users/brandonarino/Desktop/test_hourly.csv'
        self.data = pd.read_csv(self.data_path)
        self.ad = AnomalyDetection(direction='both')

    def test_from_dataframe(self):
        anoms = self.ad.from_dataframe(self.data)
        print(anoms.labeled_data)
        anoms.stl.display_plot()
        anoms.display_plot()


if __name__ == '__main__':
    unittest.main()
