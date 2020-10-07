"""
Fenics Incompressible Navier-Stokes
This script solves the laminar flow with heat transfer problem in a square duct. It has SUPG / PSPG / LSIC stabilization implemented. 
The flow is hydrodynamically developped (fully developped velocity profile imposed at inlet) and thermally developping.
It yields a Nusselt number of 4.88 compared to 6.44 in Shah & London, Laminar Flow Forced Convection in Ducts, 1978.
This difference probably comes from the meshing of the square duct: a better meshing at the wall where the temperature gradient is calculated is probably needed.
"""

from dolfin import *
from msh2xdmf import import_mesh_from_xdmf, msh2xdmf
from ufl import Min
import numpy as np

# MAKE CFD
def sim_flow(u_0, nu, cp, k, p_0, T_0, fileName, Le, He, We):
    
    # LOAD MESH
    
    mesh, boundaries, association_table = import_mesh_from_xdmf(prefix=fileName, dim=3)
    
    # Build function space
    P2 = VectorElement('Lagrange', mesh.ufl_cell() , 2)
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    element = MixedElement([P2, P1, P1])
    W = FunctionSpace(mesh, element)
    #define test and trial functions
    (v, q, s) = TestFunctions(W)
    #split functions
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

    # Define boundary conditions
    # FLOW
    # Define inflow profile from Shah & London 1976 (velocity profile in a rectangular duct)
    alpha = min(We/He, He/We) # <1
    mv = 1.7 + 0.5 * alpha**-1.4
    if alpha<=1./3.:
        nv = 2
    else:
        nv = 2 + 0.3 * (alpha - 1./3.)
    Umax = u_0 * (mv+1)/mv * (nv+1)/nv
    print("alpha, m, n, Umax = ", alpha, mv, nv, Umax)
    inflow_profile = ('0', '0', 'Umax * (1. - pow(abs(x[0]/H*2), n)) * (1. -  pow(abs(x[1]/W*2), m))')
    inflow_profile = Expression(inflow_profile, Umax=Constant(Umax), H=Constant(He), W=Constant(We), m=Constant(mv), n=Constant(nv), degree=2)
    bcu_inflow = DirichletBC(W.sub(0), inflow_profile, boundaries, association_table["inlet"])
    bcu_noslip = DirichletBC(W.sub(0), Constant((0, 0, 0)), boundaries, association_table["noslip"])
    bcu_outflow = DirichletBC(W.sub(1), Constant(p_0), boundaries, association_table["outlet"])
    bcu = [bcu_inflow, bcu_noslip, bcu_outflow]
    
    # ENERGY
    bcT_inflow = DirichletBC(W.sub(2), Constant(T_0), boundaries, association_table["inlet"])
    bcT_noslip = DirichletBC(W.sub(2), Constant(T_0+10.), boundaries, association_table["noslip"])
    bcT = [bcT_inflow, bcT_noslip]
    
    bcs = bcu + bcT
    
    # DEFINE WEAK VARIATIONAL FORM
    
    dx2 = dx(metadata={"quadrature_degree":2*3})
    
    F1 = (rho*inner(grad(u)*u, v)*dx2 +                 # Momentum advection term
        mu*inner(grad(u), grad(v))*dx2 -          # Momentum diffusion term
        inner(p, div(v))*dx2 +                    # Pressure term
        div(u)*q*dx2                            # Divergence term
    ) 
    F2 = (k*inner(grad(T), grad(s))*dx2 + # Energy advection term
        rho*cp*inner(dot(grad(T), u), s)*dx2 # Energy diffusion term
    )
    
    F = F1 + F2
  
    # SUPG / PSPG stabilization
    sigma = 2.*mu*sym(grad(u)) - p*Identity(len(u))
    # Strong formulation:
    res_strong = rho*dot(u, grad(u)) - div(sigma)
    Cinv = Constant(16*Re) # --> 16*Re is rather high, but solver diverges for lower values
    vnorm = sqrt(dot(u, u))
    tau_SUPG = Min(h**2/(Cinv*nu), h/(2.*vnorm))
    F_SUPG = inner(tau_SUPG*res_strong, rho*dot(grad(v),u) + grad(q))*dx2 # Includes PSPG
    F = F + F_SUPG
    # LSIC/grad-div:
    #tau_LSIC = rho * 2*nu/3
    tau_LSIC = h**2/tau_SUPG
    F_LSIC = tau_LSIC*div(u)*div(v)*dx2
    F = F + F_LSIC
    
    # Create VTK files for visualization output
    vtkfile_u = File('results/u.pvd')
    vtkfile_p = File('results/p.pvd')
    vtkfile_T = File('results/T.pvd')
    
    J = derivative(F, upT) # Jacobian

    problem = NonlinearVariationalProblem(F, upT, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['nonlinear_solver'] = 'newton'
    prm['newton_solver']['relaxation_parameter'] = 1.
    prm['newton_solver']['relative_tolerance'] = 1e-9
    prm['newton_solver']['absolute_tolerance'] = 1e-10
    prm['newton_solver']['maximum_iterations'] = 20
    prm['newton_solver']['error_on_nonconvergence'] = False
    prm['newton_solver']['linear_solver'] = 'mumps'
    solver.solve()
    
    # Save solution to file (VTK)
    (u, p, T) = upT.split(deepcopy=True)
    vtkfile_u << u
    vtkfile_p << p
    vtkfile_T << T
    
    
    # POST-PROCESSING
    
    def boundary_values(var, FS, boundaries, association_table, bc):
        
        vs = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, association_table[bc])), [])))
        # Get degrees of freedom associated to vertices (by indices)
        v2d = vertex_to_dof_map(VP)
        d = v2d[vs]
        nd = VP.dim()
        dim = mesh.geometry().dim()
        coordinates = VP.tabulate_dof_coordinates()
        coordinates.resize((nd, dim))
        xyz = coordinates[d]
        # Dof values
        #bnd_val = np.array(var.vector())[d]
        bnd_val = var[d]
        
        return bnd_val
   
   
    ds_bc = ds(subdomain_data=boundaries)
    
    T_in_avg = assemble(T*ds_bc(association_table["inlet"])) / (We * He)
    T_out_avg = assemble(T*ds_bc(association_table["outlet"])) / (We * He)
    DT_avg = T_out_avg - T_in_avg
    Heat_Load = u_0 * We * He * rho * cp * DT_avg
    
    htc = project(k*grad(T), VectorFunctionSpace(mesh, 'Lagrange', 1))
    Area = assemble(T/T*ds_bc(association_table["noslip"]))
    VP = FunctionSpace(mesh, 'P', 1)
    Tmin = boundary_values(np.array(T.vector()), VP, boundaries, association_table, "outlet").min()
    Tm = max(T_0+1e-3, Tmin)
    
    def DTm(T, T_0, Tm): # Log Mean Tempearature Difference is used to calculate Nu_m_T in Shah & London, Laminar Flow Forced Convection in Ducts,  1978
        return ((T - T_0) - (T - Tm)) / ln((T - T_0) / (T - Tm))
    
    htc_avg = assemble(dot(n, htc)/DTm(T,T_0,Tm)*ds_bc(association_table["noslip"])) / Area
    Nu = htc_avg * Dh / k
    Heat_Load = u_0 * We * He * rho * cp * DT_avg
    
    print('Heat_Load:', Heat_Load)
    print("htc_avg = ", htc_avg)
    print("Nu = ", Nu)
    
    return Nu


# Dimensions
Le = 18.46e-3
He = We = Dh = 2.8e-3

# Fluid properties
rho = 1.127 # air density at 20 degC, 1 atm, [kg/m3]
nu  = 16.92E-6 # air kinematic viscosity at 20 degC, 1 atm, [m2/s]
mu = nu * rho
cp = 1008. # air heat capacity @ 40°C (J/kg K) 
k = 27.35e-3 # air thermal conductivity @40°C (W/m/K) 
p_0 = 0. # outlet air pressure (atmospheric pressure), normalized
T_0 = 0. # Inlet temperature (K) 
    
fileName = "SquareDuct"

u_0 = 5.67*2 # Inlet velocity (m/s)
Re  = u_0 * Dh / nu
print("Dh = ", Dh)
print("Re = ", Re)

res = sim_flow(u_0, nu, cp, k, p_0, T_0, fileName, Le, He, We)

