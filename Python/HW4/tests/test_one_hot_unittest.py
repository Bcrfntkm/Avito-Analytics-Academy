import unittest
from one_hot_encoder import fit_transform

class TestOneHotUnittest(unittest.TestCase):
    def test_basic_sequence(self):
        result = fit_transform(['a', 'b', 'a'])
        expected = [('a', [0, 1]), ('b', [1, 0]), ('a', [0, 1])]
        self.assertEqual(result, expected)

    def test_multiple_args(self):
        result = fit_transform('x', 'y', 'x', 'z')
        expected = [('x', [0, 0, 1]), ('y', [0, 1, 0]), ('x', [0, 0, 1]), ('z', [1, 0, 0])]
        self.assertEqual(result, expected)

    def test_not_in(self):
        result = fit_transform(['p', 'q'])
        self.assertNotIn(('z', [1, 0]), result)

    def test_no_args_raises(self):
        with self.assertRaises(TypeError):
            fit_transform()

if __name__ == '__main__':
    unittest.main()