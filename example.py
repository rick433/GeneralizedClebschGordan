from transform import ClebschGordanProduct
import numpy as np

np.random.seed(42)


# helper function for printing
def pretty_print_vecs(vecs):
    for vec in vecs:
        # infer dimension of irrep by counting non-zero elements
        dim = sum(vec != 0)
        # compute rotation order from dimension of irrep, since dim = 2*l+1
        l = int((dim - 1) / 2)
        print(f"{np.round(vec[:dim], 2)} (rotation order: {l})")


num_vecs = 4
l_max = 4
rotation_orders = np.random.randint(1, l_max, num_vecs).tolist()
print(f'Generate steerable vectors of rotation orders: {rotation_orders}')
vecs = [np.random.randn(2 * a + 1) for a in rotation_orders]
# specify the Clebsch-Gordan transformation
cg_product = ClebschGordanProduct(rotation_orders)

print("Transform the tensor product of the steerable vectors")
pretty_print_vecs(vecs)
print(f"into a direct sum of steerable vectors. The individual results are:")
output = cg_product(vecs)
pretty_print_vecs(output)
