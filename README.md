# Generalized Clebsch-Gordan transformation

Simple python implementation of a generalized version of the Clebsch-Gordan transformation aka Clebsch-Gordan product aka Clebsch-Gordan decomposition.

## The vanilla Clebsch-Gordan transformation

The rotation group $SO(3)$ has an infinite number of representations given by the Wigner D-Matrices.
For each $l \in \mathbb{N}$ there is a $(2l+1) \times (2l+1)$-dimensional matrix $D^{l}(R)$ acting as a representation for
$R \in SO(3)$. Vector valued functions ${\mathbb{R}}^3 \xrightarrow{}{} {\mathbb{R}}^{2l+1}$ that transform under rotations as $f(Rx) = D^{l}(R) f(x)$ are called steerable vectors fields.
The Clebsch-Gordan transformation can be used to transform the tensor product of two steerable vector fields of rotation orders $l_1$ and $l_2$ into a set of steerable vectors with rotation orders $l \in \mathbb{N}$ such that  $| l_1 - l_2 | \leq l \leq l_1 + l_2$:

$$s^{(l_1)} \otimes s^{(l_2)} \mapsto \bigoplus_{l = | l_1 - l_2 | }^{l_1 + l_2}  q^{(l)}$$

where

$$q_m^{(l)} = \sum_{m_1 = -l_1}^{l_1} \sum_{m_2 = -l_2}^{l_2} C_{l_1 m_1 l_2 m_2}^{l m} s_{m_1}^{(l_1)} s_{m_2}^{(l_2)}$$ 

and $C_{l_1 m_1 l_2 m_2}^{l m}$ are the Clebsch-Gordan coefficients. 

## Transform more than two steerable vector fields

Since we know how the two steerable vector fields can be transformed using the Clebsch-Gordan transformation we can generalize it to more than two steerable vector fields by **iteratively applying the vanilla Clebsch-Gordan transformation**.

Thus, we can transform the tensor product of steerable vector fields $\{ s^{(l_1)}, s^{(l_2)}, ..., s^{(l_N)} \}$ of rotation orders $l_1, l_2, ...$ into a direct sum of multiple other steerable vector fields:

$$s^{(l_1)} \otimes s^{(l_2)} \otimes ... \otimes s^{(l_N)} \mapsto \bigoplus_{l} \bigoplus_{c_{l}}  q^{(l, c_l)}$$

In general, there will occur multiple new fields with the same rotation order (multiplets) which are indexed by an channel-index $c_{l}$. The number of channels might be different across different rotation orders.


This repository aims to demonstrate this transformation with a simple python implementation.  

# Python implementation

In `example.py` and in the `test` folder you can find examples on how to perform the transformation.
The general idea is as follows:

Take a list of steerable vectors of which you want to compute the Clebsch-Gordan transformation 

```python
from transform import ClebschGordanProduct
import numpy as np

# mock input: 
rotation_orders = [1,4,3,3]
vecs = [np.random.randn(2 * a + 1) for a in rotation_orders]

# specify the Clebsch-Gordan transformation
cg_product = ClebschGordanProduct(rotation_orders)

# carry out the transformation:
output = cg_product(vecs)
