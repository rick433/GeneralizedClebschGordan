import unittest
from unittest import TestCase
from sympy.physics.wigner import wigner_d
import numpy as np

np.random.seed(54)
import sys

sys.path.append('../')
from transform.transformation import ClebschGordanProduct, spherical_to_cartesian
from transform.transformation import pad_along_axis


def wigner_cartesian(l: int, alpha: float, beta: float, gamma: float) -> np.array:
    U = spherical_to_cartesian(l)
    wigner = np.array(wigner_d(l, alpha, beta, gamma)).astype(complex)
    out = U @ wigner @ U.conjugate().transpose()
    if not np.allclose(out.imag, np.zeros(out.shape)):
        print("imag part not equal zero")
    else:
        return out.real.astype(np.float32)


class TestClebschGordanProduct(TestCase):

    def _test_equivariance(self, num_vecs: int):
        rotation_orders = np.random.randint(1, 4, num_vecs).tolist()
        vecs = [np.random.randn(2 * a + 1) for a in rotation_orders]
        cg_product = ClebschGordanProduct(rotation_orders)
        mask = cg_product.mask
        winkel_test = tuple(np.random.rand(3) * 2 * np.pi)
        # get rotated inputs
        rotated_vecs = []
        for vec, rotation_order in zip(vecs, rotation_orders):
            d_matrix = wigner_cartesian(rotation_order, *winkel_test)
            rotated = d_matrix @ vec
            rotated_vecs.append(rotated)
        # f(x)
        output = cg_product(vecs)
        # f(Rx)
        output_for_rotated_input = cg_product(rotated_vecs)
        # Rf(x)
        rotated_output = []
        for channel, l in enumerate(mask):
            rotation = wigner_cartesian(l, *winkel_test)
            rotation = pad_along_axis(rotation, output.shape[-1], 0)
            rotation = pad_along_axis(rotation, output.shape[-1], 1)
            rotated = rotation @ output[channel]
            rotated_output.append(rotated)
        rotated_output = np.stack(rotated_output)
        # assert if f(Rx)=Rf(x)
        self.assertTrue(np.allclose(output_for_rotated_input, rotated_output, rtol=1e-7, atol=1e-5))
        print('Passed test!')

    def test_equivariance(self):
        for num_vecs in range(2, 5):
            for _ in range(3):
                self._test_equivariance(num_vecs)


if __name__ == '__main__':
    unittest.main()
