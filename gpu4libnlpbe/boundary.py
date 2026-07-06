import numpy
import cupy
import scipy
import fcdft
import os
import ctypes
from fcdft.solvent.pbe import KB2HARTREE
from pyscf.data.nist import BOHR
from pyscf import lib

def phi_a_finder(cost_func, jac, bottom):
    # Bisection method for a good initial guess
    phi_a = scipy.optimize.bisect(cost_func, 0.0, bottom, xtol=1e-15, maxiter=20, 
                                  full_output=False, disp=False)
    # Newton method for an accurate result
    phi_a = scipy.optimize.newton(cost_func, phi_a, fprime=jac, tol=1e-15, 
                                  maxiter=1000)
    return phi_a

def one_to_one_bc(solvent_obj, ngrids, spacing, bias, stern_sam, T, eps_sam, eps, pzc, ref_pot, jump_coeff):
    """Boundary condition generator for 1:1 electrolyte by the the Gouy-Chapman-Stern model.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        bias (float): Bias Potential.
        stern (float): Stern layer by self-assembled monolayer.
        kappa (float): Debye inverse screening length.
        T (float): Temperature.
        eps_sam (float): Dielectric constant of the self-assembled monolayer.
        eps (float): Dielectric constant of the bulk solvent.

    Returns:
        1D numpy.ndarray, 1D numpy.ndarray, float: Boundary values, electrostatic potential 
        before applying solvent-accessible surface, and the potential slope in the Stern layer.
    """
    bottom = jump_coeff * (bias - (ref_pot - pzc))
    phi_z = cupy.zeros((ngrids,)*3, dtype=cupy.float64)
    kappa = solvent_obj.kappa
    sas = solvent_obj._intermediates['sas']
    if bottom == 0.0e0:
        return phi_z.ravel()*sas, phi_z.ravel(), 0.0e0

    def cost_func(x):
        func = -2.0e0*KB2HARTREE*T*kappa*eps*numpy.sinh(-x/2.0e0/KB2HARTREE/T) - eps_sam*((bottom-x)/stern_sam)
        return func
    
    def jac(x):
        return kappa*eps*numpy.cosh(x/2.0e0/KB2HARTREE/T) + eps_sam/stern_sam

    # Jump boundary condition
    phi_a = phi_a_finder(cost_func, jac, bottom)

    # Continuous potential condition
    slope = (phi_a - bottom) / stern_sam
    z = cupy.arange(ngrids, dtype=cupy.float64) * spacing
    idx = z <= stern_sam
    phi_z[:,:,idx] = slope*z[idx]+bottom
    phi_z[:,:,~idx] = -4.0e0*KB2HARTREE*T*cupy.arctanh(cupy.exp(-kappa*(z[~idx]-stern_sam))
                                                         *cupy.tanh(-phi_a/4.0e0/KB2HARTREE/T))
    phi_z = phi_z.ravel()

    return phi_z*sas, phi_z, slope

def one_to_one_bc_grad(solvent_obj, ngrids, spacing, T, slope, phi_z):
    """Analytic gradient of the boundary conditions for 1:1 Electrolyte.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        T (float): Temperature.
        slope (float): Negative of electric field inside the Stern layer.
        phi_z (1D numpy.ndarray): Boundary value before applying the solvent-accessible surface.

    Returns:
        2D numpy.ndarray, 2D numpy.ndarray, 2D numpy.ndarray: Analytic gradient of the boundary values, 
        analytic gradient of the electrostatic potential before applying solvent-accessible surface, 
        and analytic gradient of the solvent-accessible surface.
    """
    dphidz = cupy.zeros((ngrids,)*3, dtype=cupy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    stern_sam = solvent_obj.stern_sam / BOHR
    sas = solvent_obj._intermediates['sas']
    grad_sas = solvent_obj._intermediates['grad_sas']

    kappa = solvent_obj.kappa
    z = cupy.arange(ngrids, dtype=cupy.float64) * spacing
    idx = z <= stern_sam
    dphidz[:,:,idx] = slope
    dphidz[:,:,~idx] = 2.0e0*KB2HARTREE*T*kappa*cupy.sinh(-_phi_z[:,:,~idx]/2.0e0/KB2HARTREE/T)
    dphidz = dphidz.ravel()

    grad_phi_z = cupy.zeros((ngrids**3,3), dtype=cupy.float64)
    grad_phi_z[:,2] = dphidz

    grad_bc = grad_phi_z * sas[:,None]
    grad_bc += grad_sas * phi_z[:,None]
    
    return grad_bc, grad_phi_z

def one_to_one_bc_lap(solvent_obj, ngrids, spacing, T, phi_z, grad_phi_z):
    """Laplacian of the boundary values for 1:1 Electrolyte.

    Args:
        solvent_obj (:class:`PBE`): Solvent object.
        ngrids (int): Number of grid points along each axis.
        spacing (float): Grid spacing.
        T (float): Temperature.
        phi_z (1D numpy.ndarray): Boundary value before applying the solvent-accessible surface.
        grad_phi_z (2D numpy.ndarray): Gradient of the boundary values before applying the solvent accessible surface.

    Returns:
        1D numpy.ndarray: Laplacian of the boundary values.
    """
    d2phidz2 = cupy.zeros((ngrids,)*3, dtype=cupy.float64)
    _phi_z = phi_z.reshape((ngrids,)*3)
    _grad_phi_z = grad_phi_z[:,2].reshape((ngrids,)*3)
    sas = solvent_obj._intermediates['sas']
    grad_sas = solvent_obj._intermediates['grad_sas']
    lap_sas = solvent_obj._intermediates['lap_sas']

    # Laplacian of phi(z)
    stern_sam = solvent_obj.stern_sam / BOHR
    kappa = solvent_obj.kappa
    z = cupy.arange(ngrids, dtype=cupy.float64) * spacing
    idx = z > stern_sam
    d2phidz2[:,:,idx] = -kappa*cupy.cosh(-_phi_z[:,:,idx]/2.0e0/KB2HARTREE/T)*_grad_phi_z[:,:,idx]
    d2phidz2 = d2phidz2.ravel()

    lap_bc = d2phidz2*sas + phi_z*lap_sas
    lap_bc += 2.0*(grad_phi_z * grad_sas).sum(axis=1)

    return lap_bc