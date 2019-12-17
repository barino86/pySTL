import unittest
from main.pySTL import pySTL


class TestpySTL(unittest.TestCase):

    def setUp(self):
        self.data = [2, 5, 9, 12, 8, 15, 19, 1, 6, 17]

    def test_run(self):
        period = 4
        s_window = 2

        out = pySTL(self.data, period, s_window)
        print(out)
        self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
