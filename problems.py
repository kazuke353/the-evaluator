problems = [
    {
        "name": "factorial",
        "description": "Write a function that calculates the factorial of a number.",
        "code": """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
""",
        "unit_tests": """
import unittest

class TestFactorial(unittest.TestCase):
    def test_factorial_0(self):
        self.assertEqual(factorial(0), 1)

    def test_factorial_1(self):
        self.assertEqual(factorial(1), 1)

    def test_factorial_5(self):
        self.assertEqual(factorial(5), 120)

    def test_factorial_10(self):
        self.assertEqual(factorial(10), 3628800)
"""
    },
    {
        "name": "sort_list",
        "description": "Write a function that sorts a list of numbers in ascending order.",
        "code": """
def sort_list(numbers):
    return sorted(numbers)
""",
        "unit_tests": """
import unittest

class TestSortList(unittest.TestCase):
    def test_sort_empty_list(self):
        self.assertEqual(sort_list([]), [])

    def test_sort_list_already_sorted(self):
        self.assertEqual(sort_list([1, 2, 3]), [1, 2, 3])

    def test_sort_list_reverse_sorted(self):
        self.assertEqual(sort_list([3, 2, 1]), [1, 2, 3])

    def test_sort_list_random_order(self):
        self.assertEqual(sort_list([3, 1, 4, 1, 5, 9, 2, 6]), [1, 1, 2, 3, 4, 5, 6, 9])
"""
    }
]