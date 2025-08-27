import unittest
import numpy as np
from comparison import get_first_two_clips, add_used_scores, get_score, organize_remaining_clips, get_matrix_row_sorted

matrix_data = [
                [0.99999976, 0.77272946, 0.7600999,  0.76695913],
                [0.77272946, 0.9999999,  0.8072937,  0.79386276],
                [0.7600999,  0.8072937,  1.,         0.82562786],
                [0.76695913, 0.79386276, 0.82562786, 1.        ]
        ]
matrix = np.array(matrix_data, dtype=float)

video_count = 4

class MyFunctionTests(unittest.TestCase):
    def test_get_first_two_clips(self):
        self.assertEqual(get_first_two_clips(video_count, matrix), [3, 2])

    def test_add_used_scores(self):
        used_scores = [0.82562786, 0.8072937, 0.7600999]
        row_index = 3

        self.assertEqual(add_used_scores(used_scores, matrix, row_index), [0.82562786, 0.8072937, 0.7600999, 0.82562786, 0.79386276, 0.76695913])

    def test_get_score(self):
        row_index = 3

        self.assertEqual(get_score(matrix, row_index), 0.79386276)

    def test_organize_remaining_clips(self):
        final_clips = [3, 2]
        used_scores = [0.76695913, 0.79386276, 0.82562786, 0.7600999]
        next_score = 0.8072937

        self.assertEqual(organize_remaining_clips(final_clips, 4, used_scores, matrix, next_score), [3, 2, 1, 0])
        
    def test_get_matrix_row_sorted(self):
        row_index = 3

        np.testing.assert_array_equal(get_matrix_row_sorted(matrix, row_index), [0.82562786, 0.79386276, 0.76695913])

if __name__ == '__main__':
    unittest.main()