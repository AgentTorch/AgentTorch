import torch
import unittest
from agent_torch.utils.masked_tensor import MaskedTensor


class TestMaskedTensor(unittest.TestCase):
    def setUp(self):
        self.data = torch.arange(10, dtype=torch.float32)
        self.masked_array = MaskedTensor(self.data)

    def test_integer_indexing(self):
        result = self.masked_array[5]
        self.assertEqual(result.item(), 5.0)

    def test_slice_indexing(self):
        result = self.masked_array[2:7]
        self.assertTrue(torch.all(result == self.data[2:7]))

    def test_tensor_indexing(self):
        index = torch.tensor([1, 3, 5])
        result = self.masked_array[index]
        self.assertTrue(torch.all(result == self.data[index]))

    def test_learnable_index(self):
        a = torch.tensor([3.0], requires_grad=True)
        index = a * 2
        result = self.masked_array[index]
        result.backward()
        self.assertIsNotNone(a.grad)
        self.assertEqual(a.grad.item(), 2.0)

    def test_out_of_bounds_index(self):
        with self.assertRaises(IndexError):
            _ = self.masked_array[10]

    def test_add_scalar(self):
        result = self.masked_array.add(5)
        expected = self.data + 5
        self.assertTrue(torch.all(result.data == expected))

    def test_add_tensor(self):
        other = torch.ones(10)
        result = self.masked_array.add(other)
        expected = self.data + other
        self.assertTrue(torch.all(result.data == expected))

    def test_mul_scalar(self):
        result = self.masked_array.mul(2)
        expected = self.data * 2
        self.assertTrue(torch.all(result.data == expected))

    def test_mul_tensor(self):
        other = torch.arange(10, dtype=torch.float32)
        result = self.masked_array.mul(other)
        expected = self.data * other
        self.assertTrue(torch.all(result.data == expected))


if __name__ == "__main__":
    unittest.main()
