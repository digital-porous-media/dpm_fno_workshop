from dolfin import * 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.fftpack import fft2, ifft2, fftshift

def generate_grf(N, correlation_length=10):
    kx = np.fft.fftfreq(N).reshape(-1, 1)
    ky = np.fft.fftfreq(N).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    power_spectrum = np.exp(-(k**2) * correlation_length**2)

    noise = np.random.normal(size=(N, N))
    fft_noise = fft2(noise)
    fft_field = fft_noise * np.sqrt(power_spectrum)
    field = np.real(ifft2(fft_field))
    return 50 * (field - field.min()) / (field.max() - field.min())  # normalize


realization = 5
N = 64
# 20 is number of intervals Omega is divided intoc
mesh = UnitSquareMesh(N, N)
W = FunctionSpace(mesh, "CG", 1)

# Permeability field
K_array = generate_grf(N+1, correlation_length=30)#np.exp(2 * np.random.randn(N+1, N+1))
kappa = Function(W)
kappa_vector = kappa.vector()
coords = W.tabulate_dof_coordinates().reshape((-1, 2))
x_idx = np.clip((coords[:, 0] * N).astype(int), 0, N)
y_idx = np.clip((coords[:, 1] * N).astype(int), 0, N)
kappa_vector.set_local(K_array[y_idx, x_idx])
kappa_vector.apply("insert")
# x_vals = np.linspace(0, 1, N+1)
# y_vals = np.linspace(0, 1, N+1)
# K_interp = RegularGridInterpolator((x_vals, y_vals), K_array.T)
# kappa_expr = Expression("K_interp(x[0], x[1])", K_interp=K_interp, degree=1)
# plot(mesh)

# elem = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# W = FunctionSpace(mesh, elem)


kappa_over_mu = Constant(1.0)  # physical material property
phi = Constant(0.1)  # porosity, ranging from 0 to 1
S = Constant(0.0)  # source term
v = TestFunction(W)
p = TrialFunction(W)

value_l = Constant(1.0)
value_r = Constant(0.0)

# test with differente boundary conditions...
# Imposing Dirichlet BC to the left boundary node
bc_l = DirichletBC(W, value_l, "on_boundary && near(x[0], 0)")
# bc_inj = DirichletBC(W, value_inj, "near(x[0], 0) && near(x[1], 0)")
# Imposing Dirichlet BC to the right boundary node
bc_r = DirichletBC(W, value_r, "on_boundary && near(x[0], 1)")
bcs = [bc_l, bc_r]   # list of boundary conditions to apply to the problem

F = dot(kappa*grad(p), grad(v)) * dx - S * \
    v * dx  # residual form of our equation
a, L = system(F)
ph = Function(W)  # place to store the solution
solve(a == L, ph, bcs)

# -- Convert solution to 2D NumPy array --
# Create a structured grid to sample pressure values
x_vals = np.linspace(0, 1, N+1)
y_vals = np.linspace(0, 1, N+1)
p_field = np.zeros((N+1, N+1))

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        p_field[j, i] = ph(Point(x, y))  # Note: j is y-axis, i is x-axis


# -- Plot --
plt.subplot(1, 2, 1)
plt.imshow(K_array, origin='lower', cmap='viridis', extent=[0,1,0,1])
plt.colorbar(label='Permeability')
plt.subplot(1, 2, 2)
plt.imshow(p_field, origin='lower', cmap='viridis', extent=[0,1,0,1])
plt.title("Pressure field")
plt.colorbar(label='Pressure')
plt.savefig(f"darcy_flow_result_{realization}.png")
plt.tight_layout()
plt.show()

# Get velocity field

# This is new type of element (for vector field discretization)
elem_v = VectorElement("DG", mesh.ufl_cell(), 0)
W_v = FunctionSpace(mesh, elem_v)

vf = project(-kappa * grad(ph) / phi, W_v)
fig = plt.figure()
im = plot(vf)
plt.colorbar(im, format="%.2e")
plt.savefig(f"darcy_flow_result_{realization}_vel.png")