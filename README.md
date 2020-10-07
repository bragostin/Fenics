# Fenics Incompressible Navier-Stokes
This script solves the laminar flow with heat transfer problem in a square duct. It has SUPG / PSPG / LSIC stabilization implemented. 
The flow is hydrodynamically developped (fully developped velocity profile imposed at inlet) and thermally developping.
It yields a Nusselt number of 4.88 compared to 6.44 in Shah & London, Laminar Flow Forced Convection in Ducts, 1978.
This difference may come from the meshing of the square duct: a better meshing at the wall where the temperature gradient is calculated is probably needed.

