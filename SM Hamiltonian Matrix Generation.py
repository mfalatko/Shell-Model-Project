import numpy as np
from scipy.integrate import quad
from numpy import linalg as LA
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
V0 = 51+33*((N-Z)/A)  # Potential depth [MeV]
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
