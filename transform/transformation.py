from math import sqrt
from typing import List
import numpy as np
from opt_einsum import contract
from sympy.physics.quantum.cg import CG


def spherical_to_cartesian(l: int) -> np.array:
    """
    Base change matrix (2*l+1)-dimensional cartesian basis to spherical basis
    """
    vecs = []
    for m in range(-l, l + 1):
        vec = np.zeros([2 * l + 1, 1], complex)
        if m > 0:
            vec[m + l] = -1j
            vec[-m - l - 1] = -1j * vec[m + l]
        if m < 0:
            vec[-m - l - 1] = 1j * (-1) ** m
            vec[m + l] = 1j * vec[-m - l - 1]
        if m == 0:
            vec[l] = sqrt(2)
        vecs.append(vec)
    out = 1 / sqrt(2) * np.hstack(vecs)
    return out


def ClebschG(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """
    Returns single Clebsch Gordan coefficient for the scenario where (j1,j2) couple to j3.
    """
    if float(CG(j1, m1, j2, m2, j3, m3).doit().as_real_imag()[1]) != 0:
        print(
            f"Non-Zero imaginary part in Clebsch Gordan Coefficient for (j1,m1,j2,m2,j3,m3)={j1, m1, j2, m2, j3, m3}!")
    return float(CG(j1, m1, j2, m2, j3, m3).doit().as_real_imag()[0])


def ClebschGMatrix(j1: int, j2: int, j_out: int, cartesian_basis: bool = True) -> np.array:
    """
    Returns matrix which stores CG coefficients for coupling of j1 and j2 to J_OUT
    cartesian_basis: if True, CGs are expressed in cartesian basis; If False, they are expressed in spherical basis
    """
    C = np.ones([2 * j_out + 1, int(2 * j1 + 1), int(2 * j2 + 1)], dtype=np.float64) * 0
    for m3 in range(-j_out, j_out + 1):
        for m1 in range(-j1, j1 + 1):
            for m2 in range(-j2, j2 + 1):
                C[j_out + m3, j1 + m1, j2 + m2] = ClebschG(j1, m1, j2, m2, j_out, m3)
    if not cartesian_basis:
        return C
    ##########define transformation matrices#########
    U1 = spherical_to_cartesian(j1).conjugate().transpose()
    U2 = spherical_to_cartesian(j2).conjugate().transpose()
    C = contract("Mij,in,jm->Mnm", C, U1, U2)
    U_out = spherical_to_cartesian(j_out)
    out = contract("Mij,VM->Vij", C, U_out)
    out = out.real + out.imag
    out[abs(out) < 1e-16] = 0
    return out.transpose([1, 2, 0])


def multiplicities(l_1, l_2):
    return [l for l in range(abs(l_1 - l_2), l_1 + l_2 + 1)]


class multiplet():
    def __init__(self, rotation_order, factors=None):
        self.rotation_order = rotation_order
        self.factors = factors

    def get_CG_trajectory(self, ) -> List:
        """
        returns a list of tuples with (l1,l2,l_out) that were used in the clebsch gordan decomposition to
        obtain this multiplet
        Example: to be done
        """
        if self.factors == None:
            return []
        else:
            temp = [(self.factors[0].rotation_order, self.factors[1].rotation_order, self.rotation_order)]
            out = self.factors[0].get_CG_trajectory()
            return out + temp


def all_possible_couplings(rotation_orders: list[int]) -> List:
    """
    :param _rotation_orders: list of rotation orders
    """
    _rotation_orders = rotation_orders.copy()
    if len(_rotation_orders) == 1:
        return [multiplet(rotation_order=_rotation_orders.pop(), factors=None)]
    elif len(_rotation_orders) == 2:
        l_1 = _rotation_orders[0]
        l_2 = _rotation_orders[1]
        return [multiplet(rotation_order=L, factors=[multiplet(l_1), multiplet(l_2)]) for L in multiplicities(l_1, l_2)]
    else:
        results = []
        l_2 = _rotation_orders.pop()
        right_single = multiplet(l_2, factors=None)
        left_multiplets = all_possible_couplings(rotation_orders=_rotation_orders)
        for m in left_multiplets:
            for L in multiplicities(m.rotation_order, l_2):
                results += [multiplet(rotation_order=L, factors=[m, right_single])]
        return results


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.array:
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def get_transformation_matrices(rotation_orders: list[int], pad=True):
    multiplets = all_possible_couplings(rotation_orders=rotation_orders)
    highest_output_order = max([m.rotation_order for m in multiplets])
    lowest_output_order = min([m.rotation_order for m in multiplets])
    multiplet_dict = {i: [] for i in range(lowest_output_order, highest_output_order + 1)}
    for mul in multiplets:
        multiplet_dict[mul.rotation_order].append(mul)
    trajectories = {L: [mul.get_CG_trajectory() for mul in multiplet_dict[L]] for L in multiplet_dict.keys()}
    ret = {i: [] for i in trajectories.keys()}  # will store the matrices for each L<=J
    mask = []
    for l_out in trajectories.keys():
        for traj in trajectories[l_out]:
            cg_coeffs = [ClebschGMatrix(*par) for par in traj]
            temp = cg_coeffs[0]
            for i in range(1, len(cg_coeffs)):
                # contract along the right dimensions, hard to explain here
                temp = np.tensordot(temp, cg_coeffs[i], axes=[[-1], [0]])
                # padding in order to concatenate all matrices later
            if pad:
                temp = pad_along_axis(temp, 2 * highest_output_order + 1, -1)
            ret[l_out].append(temp.astype(np.float32))
            mask.append(l_out)
    return ret, mask


class ClebschGordanProduct:

    def __init__(self, rotation_orders: list[int]):
        self.rotation_orders = rotation_orders
        matrices, mask = get_transformation_matrices(rotation_orders)
        matrices = np.vstack(list(matrices.values()))
        self._matrices = matrices
        self.mask = mask
        indices = [str(i) for i in range(len(rotation_orders))]
        rule = ",".join([",".join(indices), "B" + "".join(indices) + "L"]) + "-> BL"
        self._rule = rule

    def __call__(self, vecs: list[np.array]) -> np.array:
        return contract(self._rule, *vecs, self._matrices)
