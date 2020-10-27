from dolfin import *
from msh2xdmf import import_mesh_from_xdmf, msh2xdmf
import numpy as np


# Dimensions
Flow_Length = 18.46e-3
Dh = 2.8e-3

# Fluid properties
rho = 1.127 # air density at 20 degC, 1 atm, [kg/m3]
nu  = 16.92E-6 # air kinematic viscosity at 20 degC, 1 atm, [m2/s]
mu = nu * rho
cp = 1008. # air heat capacity @ 40°C (J/kg K) 
k = 27.35e-3 # air thermal conductivity @40°C (W/m/K) 
p_0 = 0. # outlet air pressure (atmospheric pressure), normalized
T_0 = 0. # Inlet temperature (K) 
u_0 = 5.67 # Inlet velocity (m/s)
qw = 1000. # (W/m2)
   
# LOAD MESH
mesh, boundaries, association_table = import_mesh_from_xdmf(prefix="SquareDuct", dim=3)

# Build function space
P2 = VectorElement('Lagrange', mesh.ufl_cell() , 2)
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
element = MixedElement([P2, P1, P1])
W = FunctionSpace(mesh, element)
(v, q, s) = TestFunctions(W)
upT = Function(W)

# Define initial conditions
e_u0 = Expression(('0.', '0.', 'u0'), u0=Constant(u_0), degree=1)
e_p0 = Expression('0.', degree=1)
e_T0 = Expression('0.', degree=1)
u0 = interpolate(e_u0, W.sub(0).collapse())
p0 = interpolate(e_p0, W.sub(1).collapse())
T0 = interpolate(e_T0, W.sub(2).collapse())
assign(upT, [u0, p0, T0])

(u, p, T) = split(upT)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
ds_bc = ds(subdomain_data=boundaries)

# Define boundary conditions
# Define inflow profile from Shah & London 1978 (velocity profile in a rectangular duct)
alpha = 1. # Aspect ratio
mv = 1.7 + 0.5 * alpha**-1.4
nv = 2 + 0.3 * (alpha - 1./3.)
Umax = u_0 * (mv+1)/mv * (nv+1)/nv
print("alpha, m, n, Umax = ", alpha, mv, nv, Umax)
inflow_profile = ('0', '0', 'Umax * (1. - pow(abs(x[0]/H*2), n)) * (1. -  pow(abs(x[1]/W*2), m))')
inflow_profile = Expression(inflow_profile, Umax=Constant(Umax), H=Constant(Dh), W=Constant(Dh), m=Constant(mv), n=Constant(nv), degree=2)
bcu_inflow = DirichletBC(W.sub(0), inflow_profile, boundaries, association_table["inlet"])
bcu_noslip = DirichletBC(W.sub(0), Constant((0, 0, 0)), boundaries, association_table["noslip"])
bcu_outflow = DirichletBC(W.sub(1), Constant(p_0), boundaries, association_table["outlet"])
bcu = [bcu_inflow, bcu_noslip, bcu_outflow]
# ENERGY
bcT = DirichletBC(W.sub(2), Constant(T_0), boundaries, association_table["inlet"])
bcs = bcu + [bcT]

# DEFINE WEAK VARIATIONAL FORM
F1 = (rho*inner(grad(u)*u, v)*dx +                 # Momentum ddvection term
    mu*inner(grad(u), grad(v))*dx -          # Momentum diffusion term
    inner(p, div(v))*dx +                    # Pressure term
    div(u)*q*dx                            # Divergence term
) 
F2 = (rho*cp*inner(dot(grad(T), u), s)*dx + # Energy advection term
    k*inner(grad(T), grad(s))*dx # Energy diffusion term
)
F = F1 + F2 - qw*s*ds_bc(association_table["noslip"])

J = derivative(F, upT) # Jacobian
problem = NonlinearVariationalProblem(F, upT, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps'
solver.solve()


# POST-PROCESSING
Flow_Area = Dh**2
Heat_Load = 4. * qw * Flow_Length * Dh
print("Heat_Load, qw = ", Heat_Load, qw)

def Tm(z):
    Tm = T_0 + Heat_Load * z / Flow_Length / (u_0 * Flow_Area * rho * cp)
    return Tm

x = SpatialCoordinate(mesh)
#Area = assemble(Constant(1.)*ds_bc(association_table["noslip"])) # Error: ufl.log.UFLException: This integral is missing an integration domain.
Area = assemble(T/T*ds_bc(association_table["noslip"])) # There should be a simpler way than integrating T/T

# Sanity check
Tm_avg = assemble(Tm(x[2])*ds_bc(association_table["noslip"])) / Area
print("Tm_avg assemble = ", Tm_avg) # this should be equal to DTm_avg mean
Tm_avg = (Tm(0.) + Tm(Flow_Length)) / 2
print("Tm_avg mean = ", Tm_avg)

htc_avg = assemble(dot(n, k*grad(T))/(T-Tm(x[2]))*ds_bc(association_table["noslip"])) / Area
print("htc_avg Tm(x[2]) = ", htc_avg)
