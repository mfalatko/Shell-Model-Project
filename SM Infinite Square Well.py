import numpy as np
from scipy.integrate import quad
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set()
sns.set_style('darkgrid')

# Record start time
start_time = time.time()

# Constants
ħ = 6.582119569e-22  # Reduced Planck's constant [MeV.s]
m = 938.272  # Proton mass [MeV/c^2]
Z = 2  # Atomic number
A = 4  # Mass number
N = A - Z

# Woods-Saxon potential parameters
V0 = 51+33*((N-Z)/A)  # Potential depth [keV]
a = 0.67  # Diffuseness [fm]
r0 = 1.27  # Set value [fm]
R = r0 * A**(1/3)  # Radius [fm]

# Coulomb potential parameters
e = 1.602176634e-19  # Elementary charge [C]
ε0 = 8.854187817e-27  # Vacuum permittivity [F/fm]

# Define the Woods-Saxon potential
def woods_saxon(r):
    return -V0 / (1 + np.exp((r - R) / a))

# Coulomb potential
def coulomb_potential(r):
    return Z * e**2 / (4 * np.pi * ε0 * r) * (3/2 - 1/2 * (r/R)**2)

# Kinetic energy operator
def kinetic_energy(psi, r):
    return -ħ**2 / (2 * m) * (psi(r + 1e-5) - 2 * psi(r) + psi(r - 1e-5)) / (1e-5)**2

# Hamiltonian matrix element integrand
def hamiltonian_integrand(r, n, m):
    psi_n = lambda x: np.sqrt(2 / R) * np.sin(n * np.pi * x / R)
    psi_m = lambda x: np.sqrt(2 / R) * np.sin(m * np.pi * x / R)
    return kinetic_energy(psi_n, r) + kinetic_energy(psi_m, r) + woods_saxon(r) * psi_n(r) * psi_m(r) + coulomb_potential(r) * psi_n(r) * psi_m(r)

# Hamiltonian matrix element
def hamiltonian_element(n, m):
    result, _ = quad(lambda r: hamiltonian_integrand(r, n, m), 0, R)
    return result

# Size of the matrix (considering a finite basis set)
matrix_size = 4

# Build Hamiltonian matrix
hamiltonian_matrix = np.zeros((matrix_size, matrix_size))
for i in range(matrix_size):
    for j in range(matrix_size):
        hamiltonian_matrix[i, j] = hamiltonian_element(i + 1, j + 1)

# Print Hamiltonian matrix
print("Hamiltonian Matrix:")
print(hamiltonian_matrix)


# Lancosz Algorithm taken and adapted from
# @misc{zachtheyek_2022,
#  author = {Yek, Xi},
#  title = {Lanczos-Algoritm},
#  year = {2022},
#  publisher = {GitHub},
# journal = {GitHub repository},
#  url = {\url{https://github.com/zachtheyek/Lanczos-Algorithm}},
#  commit = {233eb02} }


# Function to tri-diagonalize a matrix
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

# Lanczos algorithm
def lanczos(A, v1):
    np.set_printoptions(precision=3, suppress=True)
    # First iteration steps
    x, y = [], []
    n = A.shape[1]
    v2, beta = 0.0, 0.0

    for i in range(n):
        # Iteration steps
        w_prime = np.dot(A, v1)
        conj = np.matrix.conjugate(w_prime)
        alpha = np.dot(conj, v1)
        w = w_prime - alpha * v1 - beta * v2
        beta = np.linalg.norm(w)
        x.append(np.linalg.norm(alpha))

        # Reset
        if i < (n - 1):
            y.append(beta)
        v2 = v1
        v1 = w / beta

    return tridiag(y, x, y)

# Initial vector for Lanczos algorithm
n = hamiltonian_matrix.shape[0]
v_0 = np.zeros(n)
v_0.fill(1.)
v = v_0 / np.linalg.norm(v_0)

# Obtaining the tri-diagonal matrix T using the Hamiltonian matrix
T = lanczos(hamiltonian_matrix, v)
print(f'\nTridiagonalization of the Hamiltonian Matrix: \n{T}')

# Finding the eigenvalues w and eigenvectors v of the tri-diagonal matrix
w, v = LA.eig(T)

# Print eigenvalues in keV directly
print(f'\nAssociated eigenvalues (in keV): \n{w}')
print(f'\nAssociated eigenvectors: \n{v}')

# Record end time
end_time = time.time()

# Print total runtime
total_runtime = end_time - start_time
print(f"\nTotal Runtime: {total_runtime} seconds\n")
