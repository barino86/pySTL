import unittest
from stl import STLDecomposition


class TestpySTL(unittest.TestCase):

    def setUp(self):
        self.data = [2, 5, 9, 12, 8, 15, 19, 1, 6, 17]
        self.period = 4
        self.s_window = 2

        self.seasonal = [-3.6846972069291994, -0.05666929305924667, -0.1446486044010491, 1.8204205167156722,
                         -2.920572920265397, 3.697780786186189, 7.594950153900969, -10.38689232099828,
                         -3.931557101306311, 5.232254332002869]
        self.trend = [6.673328842628257, 7.367571688448032, 8.061814534267807, 8.756057380087583,
                      9.385721300412259, 10.015385220736936, 10.645049141061612, 11.202924448454386,
                      11.76079975584716, 12.318675063239933]
        self.remainder = [-0.9886316356990577, -2.3109023953887853, 1.0828340701332415, 1.4235221031967455,
                          1.5348516198531374, 1.2868339930768737, 0.7600007050374185, 0.18396787254389402,
                          -1.829242654540849, -0.550929395242802]

    def test_run_new(self):
        stl = STLDecomposition(series=self.data, period=self.period, s_window=self.s_window).stl()
        print(stl)
        print(stl.data)
        self.assertIsNotNone(stl.data)
        self.assertEqual(list(stl.data['seasonal']), self.seasonal)
        self.assertEqual(list(stl.data['trend']), self.trend)
        self.assertEqual(list(stl.data['remainder']), self.remainder)

    def test_run_new_periodic(self):
        stl = STLDecomposition(series=self.data, period=self.period, periodic=True).stl()
        print(stl.data)

        fig = stl.build_plot()
        stl.display_plot(fig)


if __name__ == '__main__':
    unittest.main()
