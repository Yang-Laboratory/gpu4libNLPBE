import numpy
import cupy
from pyscf import lib
from pyscf.data.nist import BOHR
from fcdft.solvent.pbe import M2HARTREE, KB2HARTREE

PI = numpy.pi

def _one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    """Size-modified 1:1 ion charge density.

    Args:
        solvent_obj (:class:`PBE`): Solvent object
        phi_tot (1D numpy.ndarray, optional): Total electrostatic potential. Defaults to None.
        cb (float, optional): Ion concentration in atomic unit. Defaults to None.
        lambda_r (1D numpy.ndarray, optional): Ion-exclusion function. Defaults to None.
        T (float, optional): Temperature. Defaults to None.

    Returns:
        1D numpy.ndarray: Ion charge density
    """
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    lambda_r = cupy.asarray(lambda_r)

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74e0 / (4.0e0/3.0e0 * PI * (cation_rad**3 + anion_rad**3))
    if cb == 0.0e0:
        return cupy.zeros_like(phi_tot)

    x = phi_tot / KB2HARTREE / T

    mask = abs(x) < 691.4
    sinh = cupy.full_like(x, 1.0e300) * cupy.sign(x)
    cosh = cupy.full_like(x, 1.0e300)
    sinh[mask] = cupy.sinh(x[mask])
    cosh[mask] = cupy.cosh(x[mask])

    rho_ions = -2.0 * lambda_r * cb * sinh / (1.0 - cb/c12 + cb/c12 * lambda_r * cosh)

    return cupy.asarray(rho_ions)