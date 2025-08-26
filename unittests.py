import unittest
import numpy as np
from comparison import get_first_two_clips, add_used_scores, get_score

class MyFunctionTests(unittest.TestCase):
    # def test_get_first_two_clips(self):
    #     matrix_data = [
    #                     [0.99999976, 0.77272946, 0.7600999,  0.76695913],
    #                     [0.77272946, 0.9999999,  0.8072937,  0.79386276],
    #                     [0.7600999,  0.8072937,  1.,         0.82562786],
    #                     [0.76695913, 0.79386276, 0.82562786, 1.        ]
    #             ]
    #     matrix = np.array(matrix_data, dtype=float)
    #     video_count = 4
    #     self.assertEqual(get_first_two_clips(video_count, matrix), [3, 2])

    # def test_add_used_scores(self):
    #     matrix_data = [
    #                     [0.99999976, 0.77272946, 0.7600999,  0.76695913],
    #                     [0.77272946, 0.9999999,  0.8072937,  0.79386276],
    #                     [0.7600999,  0.8072937,  1.,         0.82562786],
    #                     [0.76695913, 0.79386276, 0.82562786, 1.        ]
    #             ]
    #     matrix = np.array(matrix_data, dtype=float)
    #     used_scores = []
    #     row_index = 3

    #     self.assertEqual(add_used_scores(used_scores, matrix, row_index), [0.82562786, 0.79386276, 0.76695913])

    def test_get_score(self):
        matrix_data = [
                        [0.99999976, 0.77272946, 0.7600999,  0.76695913],
                        [0.77272946, 0.9999999,  0.8072937,  0.79386276],
                        [0.7600999,  0.8072937,  1.,         0.82562786],
                        [0.76695913, 0.79386276, 0.82562786, 1.        ]
                ]
        matrix = np.array(matrix_data, dtype=float)
        row_index = 3

        self.assertEqual(get_score(matrix, row_index), 0.79386276)

if __name__ == '__main__':
    unittest.main()