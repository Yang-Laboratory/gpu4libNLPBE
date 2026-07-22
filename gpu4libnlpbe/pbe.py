import os
import ctypes
import numpy
import cupy
import cupyx
import scipy
import h5py

from pyscf.data.nist import *
from pyscf.lib import logger
from gpu4pyscf.lib import logger as gpulogger
from libnlpbe import pbe
from pyscf import lib
import gpu4libnlpbe
from pyscf import df
from pyscf import gto
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import pack_tril
from gpu4pyscf.dft import numint as _ni
from gpu4pyscf.gto.mole import cart2sph_by_l
from gpu4pyscf.scf.int4c2e import libgint, libgvhf
from gpu4pyscf.df.int3c2e import make_fake_mol, get_pairing, get_ao_pairs
from gpu4pyscf.df.int3c2e import VHFOpt
from gpu4pyscf.df import int3c2e_bdiv
from cupyx.scipy.linalg import solve_triangular


PI = numpy.pi
KB2HARTREE = BOLTZMANN / HARTREE2J
M2HARTREE = AVOGADRO*BOHR**3*1.e-27

libamgcl = lib.load_library(os.path.join(gpu4libnlpbe.__path__[0], 'lib', 'libdinmh.so'))
libamgcl.amg_create.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int, # tot_ngrids
    ctypes.c_void_p, # indptr
    ctypes.c_void_p, # indices
    ctypes.c_void_p, # data
    ctypes.c_int, # max_levels
    ctypes.c_int, # coarse_enough
    ctypes.c_int, # relax
    ctypes.c_int, # npre
    ctypes.c_int, # npost
    ctypes.c_int, # ncycle
    ctypes.c_int # pre_cycles
]
libamgcl.amg_create.restype = ctypes.c_void_p
libamgcl.amg_vcycle.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int
]
libamgcl.csr_matvec.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_void_p,  # indptr
    ctypes.c_void_p,  # indices
    ctypes.c_void_p,  # data
    ctypes.c_void_p,  # x
    ctypes.c_void_p,   # out  
    ctypes.c_int,     # tot_ngrids
    ctypes.c_int      # nnz
]
libamgcl.csr_matvec.restype = None

make_gradient_matrix = pbe.make_gradient_matrix

def gradient(solvent_obj, phi, ngrids, spacing):
    """8th-order finite-difference gradient of a scalar cupy field.

    Returns an ``(n_grid, 3)`` cupy array. Used at convergence to build the
    polarization charge density ``rho_pol``; matches the stencil used to
    assemble the linear operator in :func:`make_operator`.
    """
    grad = make_gradient_matrix(ngrids)
    I = scipy.sparse.identity(ngrids, format='csr')
    G = (scipy.sparse.kron(grad, scipy.sparse.kron(I, I)),
         scipy.sparse.kron(I, scipy.sparse.kron(grad, I)),
         scipy.sparse.kron(I, scipy.sparse.kron(I, grad)))
    dphi = cupy.empty((phi.size, 3), dtype=cupy.float64)
    for xi in range(3):
        Gx = cupyx.scipy.sparse.csr_matrix(G[xi])
        dphi[:, xi] = Gx.dot(phi) / spacing
    return dphi

def _basis_seg_contraction(mol, allow_replica=1, sparse_coeff=False):
    # from gpu4pyscf.gto.mole import basis_seg_contraction
    _bas = mol._bas
    if _bas.size > 0 and (_bas[:, gto.NCTR_OF] == 1).all():
        pmol = mol.copy()
        pmol.cart = True
        if sparse_coeff:
            return pmol, None
    raise RuntimeError('Consult with the developer.')

def _sort_mol(mol0, cart=True):
    """Replacing deepcopy with copy.
    Sort mol._bas with respect to orbital angular momentum.

    Args:
        mol0 (_type_): _description_
        cart (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    mol = mol0.copy()
    l_ctrs = mol0._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
    uniq_l_ctr, _, inv_idx, l_ctr_counts = numpy.unique(
        l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)
    
    sorted_idx = numpy.argsort(inv_idx.ravel(), kind='stable').astype(numpy.int32)

    # Sort basis inplace
    mol._bas = mol0._bas[sorted_idx]
    return mol, sorted_idx, uniq_l_ctr, l_ctr_counts

def aux_ao_idx(auxmol, sorted_aux_idx):
    aux_loc = auxmol.ao_loc_nr(cart=auxmol.cart)
    shell_sizes = (aux_loc[1:] - aux_loc[:-1]).astype(numpy.int64)
    sorted_starts = aux_loc[:-1].astype(numpy.int64)[sorted_aux_idx]
    sorted_sizes = shell_sizes[sorted_aux_idx]
    total = int(sorted_sizes.sum())
    out_starts = numpy.concatenate(([0], numpy.cumsum(sorted_sizes[:-1])))
    src = numpy.repeat(sorted_starts, sorted_sizes)
    intra = numpy.arange(total, dtype=numpy.int64) - numpy.repeat(out_starts, sorted_sizes)
    return src + intra    

def _build_aux(self, cutoff):
    """Build the bra part of (P|delta)

    Args:
        cutoff (_type_): _description_
    """
    _mol = self.mol
    mol = _basis_seg_contraction(_mol, allow_replica=True, sparse_coeff=True)[0]

    _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = _sort_mol(mol)

    self.nctr = len(uniq_l_ctr)
    self.l_ctr_counts = l_ctr_counts

    # sort fake mol
    fake_mol = make_fake_mol()
    _, _, fake_uniq_l_ctr, fake_l_ctr_counts = _sort_mol(fake_mol)

    # Initialize vhfopt after reordering mol._bas
    _vhf.VHFOpt.__init__(self, _sorted_mol, self._intor, self._prescreen,
                         self._qcondname, self._dmcondname)
    self.direct_scf_tol = cutoff

    # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
    q_cond = self.get_q_cond()
    l_ctr_offsets = numpy.append(0, numpy.cumsum(l_ctr_counts))
    log_qs, pair2bra, pair2ket = get_pairing(
        l_ctr_offsets, l_ctr_offsets, q_cond, 
        diag_block_with_triu=True, aosym=True)
    self.log_qs = log_qs.copy()

    # contraction coefficient for ao basis
    cart_ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    sph_ao_loc = _sorted_mol.ao_loc_nr(cart=False)
    self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
    self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
    self.angular = [l[0] for l in uniq_l_ctr]

    # Sorted AO indices
    self._ao_idx = aux_ao_idx(_mol, sorted_idx).astype(numpy.int32)

    ao_loc = _sorted_mol.ao_loc_nr(cart=_mol.cart)
    self.ao_pairs_row, self.ao_pairs_col = get_ao_pairs(pair2bra, pair2ket, ao_loc)

    # Sorted (P|delta) pairing
    nbas_aux = _sorted_mol.nbas
    fake_shell = nbas_aux
    bra_pair2bra, bra_pair2ket, bra_log_qs = [], [], []
    for p0, p1 in zip(l_ctr_offsets[:-1], l_ctr_offsets[1:]):
        bra_pair2bra.append(numpy.arange(p0, p1, dtype=numpy.int32))
        bra_pair2ket.append(fake_shell * numpy.ones(p1-p0, dtype=numpy.int32))
        bra_log_qs.append(numpy.ones(p1 - p0, dtype=numpy.int32))

    self._aux_mol = mol
    self._sorted_mol = _sorted_mol
    self._fake_mol = fake_mol
    self._fake_l_ctr_counts = fake_l_ctr_counts
    self._l_ctr_offsets = l_ctr_offsets
    self._nbas_aux = nbas_aux
    self._fake_shell = fake_shell
    self._mol_pair2bra = pair2bra
    self._mol_pair2ket = pair2ket
    self._bra_pair2bra = bra_pair2bra
    self._bra_pair2ket = bra_pair2ket
    self._bra_log_qs = bra_log_qs
    self._n_aux_groups = len(bra_log_qs)
    self.nao_cart = _sorted_mol.nao
    self.aosym = True

def _build_grids(self, coords, expnt=1e16):
    """Build the ket part of (P|delta) and complete VHFOpt build

    Args:
        coords (_type_): _description_
        expnt (_type_, optional): _description_. Defaults to 1e16.
    """
    fakemol = gto.fakemol_for_charges(coords, expnt=expnt)
    auxmol = _basis_seg_contraction(fakemol, allow_replica=True, sparse_coeff=True)[0]
    
    # sort auxiliary mol
    _sorted_auxmol, sorted_aux_idx, aux_uniq_l_ctr, aux_l_ctr_counts = _sort_mol(auxmol)
    self.aux_l_ctr_counts = aux_l_ctr_counts

    _tot_mol = self._sorted_mol + self._fake_mol + _sorted_auxmol
    _tot_mol.cart = True

    self._tot_mol = _tot_mol
    self._sorted_auxmol = _sorted_auxmol

    aux_l_ctr_offsets = numpy.append(0, numpy.cumsum(aux_l_ctr_counts))
    cart_aux_loc = _sorted_auxmol.ao_loc_nr(cart=True)
    sph_aux_loc = _sorted_auxmol.ao_loc_nr(cart=False)
    self.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
    self.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
    self.aux_angular = [l[0] for l in aux_uniq_l_ctr]
    self._aux_ao_idx = aux_ao_idx(fakemol, sorted_aux_idx).astype(numpy.int64)

    # tot_mol = mol (nbas) + fake_mol (1) + auxmol (nbatch)
    # Pairing fake_mol and auxmol
    grid_base = self._nbas_aux + 1
    fake_shell = self._fake_shell
    grid_pair2bra, grid_pair2ket, grid_log_qs = [], [], []
    for p0, p1 in zip(aux_l_ctr_offsets[:-1], aux_l_ctr_offsets[1:]):
        grid_pair2bra.append(numpy.arange(p0 + grid_base, p1 + grid_base, dtype=numpy.int32))
        grid_pair2ket.append(fake_shell * numpy.ones(p1-p0, dtype=numpy.int32))
        grid_log_qs.append(numpy.ones(p1 - p0, dtype=numpy.int32))
    self._n_grid_groups = len(grid_log_qs)

    # Full pairing list
    pair2bra = self._mol_pair2bra + self._bra_pair2bra + grid_pair2bra
    pair2ket = self._mol_pair2ket + self._bra_pair2ket + grid_pair2ket
    self.pair2bra = pair2bra
    self.pair2ket = pair2ket
    self.aux_log_qs = (self._bra_log_qs + grid_log_qs)

    self.bas_pairs_locs = numpy.append(
        0, numpy.cumsum([x.size for x in pair2bra])).astype(numpy.int32)
    ncptype = len(self.log_qs)
    self.cp_idx, self.cp_jdx = numpy.tril_indices(ncptype)

    # Collect GPU memory
    if getattr(self, '_bpcache', None):
        for device_id, bpcache in self._bpcache.items():
            libgvhf.GINTdel_basis_prod(ctypes.byref(bpcache))
    self._bpcache = {}

def int2e_grids(intopt, coords, cK, out, off, charge_exponents=1e16, direct_scf_tol=1e-14):
    # Build grid part
    _build_grids(intopt, coords, expnt=charge_exponents)

    auxmol = intopt.mol
    nao_cart = intopt.nao_cart
    n_aux_groups = intopt._n_aux_groups
    n_log_qs = len(intopt.log_qs)

    # mol
    cart_ao_loc = numpy.asarray(intopt.cart_ao_loc, dtype=numpy.int32)
    sph_ao_loc = numpy.asarray(intopt.sph_ao_loc, dtype=numpy.int64)
    # auxmol
    cart_aux_loc = numpy.asarray(intopt.cart_aux_loc, dtype=numpy.int32)
    sph_aux_loc = numpy.asarray(intopt.sph_aux_loc, dtype=numpy.int64)

    ao_idx = cupy.asarray(intopt._ao_idx.astype(numpy.int64))
    grid_ao_idx = intopt._aux_ao_idx

    norb_cart = nao_cart + 1 + intopt._sorted_auxmol.nao
    stream = cupy.cuda.get_current_stream()
    log_cutoff = numpy.log(direct_scf_tol)
    omega = 0.0

    # Sorted df coefficients
    cK_sorted = cK[ao_idx]

    for k_id in range(n_aux_groups):
        lk = intopt.angular[k_id]
        k0, k1 = cart_ao_loc[k_id], cart_ao_loc[k_id + 1]
        nk = k1 - k0
        aux_idx = ao_idx[sph_ao_loc[k_id]:sph_ao_loc[k_id + 1]]
        cK_k = cK_sorted[sph_ao_loc[k_id]:sph_ao_loc[k_id + 1]]

        bins_locs_k = numpy.array([0, len(intopt._bra_log_qs[k_id])], dtype=numpy.int32)
        bins_floor_k = numpy.array([100], dtype=numpy.double)
        cp_k_id = n_log_qs + k_id

        for l_id in range(intopt._n_grid_groups):
            l0, l1 = cart_aux_loc[l_id], cart_aux_loc[l_id + 1]
            nl = l1 - l0
            bins_locs_l = numpy.array([0, len(intopt.aux_log_qs[n_aux_groups + l_id])], dtype=numpy.int32)
            bins_floor_l = numpy.array([100], dtype=numpy.double)
            cp_l_id = n_log_qs + n_aux_groups + l_id

            int2c_blk = cupy.zeros((nk, nl), dtype=cupy.float64, order='F')
            ao_offsets = numpy.array([k0, nao_cart,
                                      nao_cart + 1 + l0, nao_cart], dtype=numpy.int32)
            strides = numpy.array([1, nao_cart, nk, nao_cart * nao_cart], dtype=numpy.int32)
            err = libgint.GINTfill_int2e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(int2c_blk.data.ptr, ctypes.c_void_p),
                ctypes.c_int(norb_cart),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_k.ctypes.data_as(ctypes.c_void_p),
                bins_locs_l.ctypes.data_as(ctypes.c_void_p),
                bins_floor_k.ctypes.data_as(ctypes.c_void_p),
                bins_floor_l.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(bins_locs_k) - 1),
                ctypes.c_int(len(bins_locs_l) - 1),
                ctypes.c_int(cp_k_id),
                ctypes.c_int(cp_l_id),
                ctypes.c_double(log_cutoff),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError("int2c2e failed\n")

            if not auxmol.cart:
                c2s = cart2sph_by_l(lk)
                nshl = nk // c2s.shape[0]
                t_cart = cupy.ascontiguousarray(int2c_blk).reshape(nshl, c2s.shape[0], nl)
                int2c_blk = cupy.einsum('min,ip->mpn', t_cart, c2s, optimize=True)
                int2c_blk = int2c_blk.reshape(nshl * c2s.shape[1], nl)

            grid_idx = cupy.asarray(grid_ao_idx[sph_aux_loc[l_id]:sph_aux_loc[l_id + 1]])
            contrib = int2c_blk.T.dot(cK_k)
            cupyx.scatter_add(out, off + grid_idx, contrib)

    stream.synchronize()

def make_phi_sol(solvent_obj, dm=None, coords=None):
    """GPU-accelerated solute electrostati potential based on density fitting.

    Args:
        solvent_obj (_type_): _description_
        dm (_type_, optional): _description_. Defaults to None.
        coords (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if dm is None:
        dm = solvent_obj._dm
    if coords is None:
        coords = solvent_obj.grids.coords

    verbose = solvent_obj.verbose
    log = gpulogger.new_logger(solvent_obj, verbose)
    t0 = log.init_timer()
    mol = solvent_obj.mol
    tot_ngrids = coords.shape[0]

    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    dist[dist < 1.0e-100] = numpy.inf
    Vnuc = numpy.tensordot(1.0e0 / dist, Z, axes=([0], [0]))

    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    dms = numpy.asarray(dm.real)

    verbose = mol.verbose
    mol.verbose = 0
    auxmol = df.addons.make_auxmol(mol)
    mol.verbose = verbose

    nbatch = 256*256
    j2c = auxmol.intor('int2c2e')
    j2c = cupy.asarray(j2c)    
    rhoj = int3c2e_bdiv.contract_int3c2e_dm(mol, auxmol, dms)
    cd_low = cupy.linalg.cholesky(j2c)
    y = solve_triangular(cd_low, rhoj, lower=True, overwrite_b=True)
    d = solve_triangular(cd_low, y, trans='T', lower=True, overwrite_b=True)

    fakemol = gto.fakemol_for_charges(coords[:1])
    intopt = VHFOpt(auxmol, fakemol, 'int2e')
    # Build the auxmol part
    _build_aux(intopt, 1e-14)

    Vele = cupy.zeros(tot_ngrids)
    for p0, p1 in lib.prange(0, tot_ngrids, nbatch):
        int2e_grids(intopt, coords[p0:p1], d, Vele, p0, direct_scf_tol=1e-14)
    Vele = Vele.get()

    MEP = Vnuc - Vele
    t0 = log.timer('phi_sol', *t0)
    return lib.tag_array(MEP, Vnuc=Vnuc, Vele=-Vele)

def rho_ions_one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
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

def drho_ions_one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    """drho_ions / dphi"""
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    if cb == 0.0:
        return cupy.zeros_like(phi_tot)

    lambda_r = cupy.asarray(lambda_r)
    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74 / (4.0 / 3.0 * PI * (cation_rad**3 + anion_rad**3))

    x = phi_tot / KB2HARTREE / T
    mask = abs(x) < 691.4
    drho_ions = cupy.zeros_like(x)
    cosh = cupy.cosh(x[mask])
    a = cb / c12 * lambda_r[mask]
    b = 1.0 - cb / c12
    D = b + a * cosh
    # / D / D instead of D**2 to avoid unnecessary value overflow error msg
    drho_ions[mask] = (-2.0 * lambda_r[mask] * cb) * (b * cosh + a) / D / D
    drho_ions /= KB2HARTREE * T
    return drho_ions


def make_operator(solvent_obj, grad_lneps=None):
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    if grad_lneps is None:
        _intermediates = solvent_obj._intermediates
        eps = _intermediates['eps']
        grad_eps = _intermediates['grad_eps']
        grad_lneps = grad_eps / eps[:,None]

    L = solvent_obj.L # nabla**2 = -L / spacing**2
    # A = nabla ln(eps) * nabla - L / spacing**2
    if solvent_obj.operator is None:
        grad = make_gradient_matrix(ngrids)
        I = scipy.sparse.identity(ngrids, format='csr')
        G = (scipy.sparse.kron(grad, scipy.sparse.kron(I, I)),
             scipy.sparse.kron(I, scipy.sparse.kron(grad, I)),
             scipy.sparse.kron(I, scipy.sparse.kron(I, grad)))
        A = -L / spacing**2
        for xi in range(3):
            A = A + scipy.sparse.diags(grad_lneps[:, xi]) @ (G[xi] / spacing)
        solvent_obj.operator = cupyx.scipy.sparse.csr_matrix(A)

    A = solvent_obj.operator
    handle = solvent_obj.handle

    def apply_A(v, out):
        libamgcl.csr_matvec(handle,
                            A.indptr.data.ptr,
                            A.indices.data.ptr,
                            A.data.data.ptr,
                            v.data.ptr,
                            out.data.ptr,
                            ctypes.c_int(tot_ngrids),
                            ctypes.c_int(A.nnz))
        return out
    
    return apply_A

def make_precond(solvent_obj, drho_ions_scr=None):
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    L = solvent_obj.L
    precond = L / spacing**2 + scipy.sparse.diags(drho_ions_scr.get(), format='csr')

    if solvent_obj.hierarchy is None:
        verbose = solvent_obj.verbose
        log = gpulogger.new_logger(solvent_obj, verbose)
        t0 = log.init_timer()
        handle = solvent_obj.handle
        hierarchy = libamgcl.amg_create(handle,
                                       ctypes.c_int(tot_ngrids),
                                       precond.indptr.ctypes.data,
                                       precond.indices.ctypes.data,
                                       precond.data.ctypes.data,
                                       ctypes.c_int(4),
                                       ctypes.c_int(1000),
                                       ctypes.c_int(0),
                                       ctypes.c_int(1),
                                       ctypes.c_int(1),
                                       ctypes.c_int(1),
                                       ctypes.c_int(1))
        solvent_obj.hierarchy = hierarchy
        t0 = log.timer('make_precond', *t0)
    hierarchy = solvent_obj.hierarchy

    def vcycle(r, out):
        libamgcl.amg_vcycle(ctypes.c_void_p(hierarchy),
                            r.data.ptr,
                            out.data.ptr,
                            ctypes.c_int(tot_ngrids))
        return out

    return vcycle

def _release_caches(solvent_obj):
    """Release memory after make_phi"""
    if getattr(solvent_obj, 'hierarchy', None) is not None:
        hierarchy = solvent_obj.hierarchy
        libamgcl.amg_destroy(ctypes.c_void_p(hierarchy))
    libamgcl._cusparseDestroy(solvent_obj.handle)
    solvent_obj.hierarchy = None

def make_phi(solvent_obj, phi_sol=None, rho_sol=None):
    if solvent_obj._intermediates is None: solvent_obj.build()
    _intermediates = solvent_obj._intermediates

    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = solvent_obj.grids.get_ngrids()
    T = solvent_obj.T
    spacing = solvent_obj.grids.spacing
    cb = solvent_obj.cb * M2HARTREE

    eps = _intermediates['eps']
    lambda_r = _intermediates['lambda_r']
    grad_eps = _intermediates['grad_eps']

    max_cycle = solvent_obj.max_cycle # Newton cycle
    inner_max_cycle = solvent_obj.inner_max_cycle

    C_TST = 0.01
    p_TST = 1.0

    grad_lneps = grad_eps / eps[:, None]
    get_rho_ions = solvent_obj._gen_get_rho_ions()
    get_drho_ions = solvent_obj._gen_drho_ions()

    bc, const_src = solvent_obj._boundary_conditions(ngrids, spacing)

    inv_eps = cupy.array(4.0 * PI / eps)

    A = solvent_obj.make_operator(grad_lneps)

    if cb == 0.0:
        drho_ions_scr = cupy.zeros(tot_ngrids)
    else:
        drho_ions_scr = -inv_eps * get_drho_ions(solvent_obj, cupy.zeros(tot_ngrids), cb, lambda_r, T)

    precond = solvent_obj.make_precond(drho_ions_scr)

    rho_sol = cupy.asarray(rho_sol)
    grad_lneps = cupy.asarray(grad_lneps)
    eps = cupy.asarray(eps)

    def residual(phi_opt, out):
        phi_tot = phi_opt + bc
        rho_ions = get_rho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        if cupy.isnan(rho_ions).any():
            return None, rho_ions
        rho_tot = rho_sol + rho_ions
        A(phi_opt, out)
        out += inv_eps * rho_tot + const_src
        return out, rho_ions

    def finalize(phi_opt, rho_ions):
        rho_tot = rho_sol + rho_ions
        dphi = solvent_obj.gradient(phi_opt, ngrids, spacing)
        rho_iter = 0.25 / PI * (grad_lneps * dphi).sum(axis=1)
        rho_pol = (1.0 - eps) / eps * rho_tot + rho_iter
        phi_tot = phi_opt + bc
        return phi_tot.get(), rho_ions.get(), rho_pol.get()

    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info(solvent_obj, ' |  Poisson-Boltzmann Solver with the Multigrid Scheme  |')
    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

    phi_opt = cupy.zeros(tot_ngrids, dtype=cupy.float64)
    res_old = cupy.empty(tot_ngrids, dtype=cupy.float64)
    res_new = cupy.empty(tot_ngrids, dtype=cupy.float64)
    res_outer, rho_ions = residual(phi_opt, res_new)

    if res_outer is None:
        logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
        return None, None, None

    fnorm = cupy.linalg.norm(res_outer)

    res_inner = cupy.empty(tot_ngrids, dtype=cupy.float64)
    dv = cupy.empty(tot_ngrids, dtype=cupy.float64)
    Av = cupy.empty(tot_ngrids, dtype=cupy.float64)
    b = cupy.empty(tot_ngrids, dtype=cupy.float64)

    t0 = (logger.process_clock(), logger.perf_counter())

    cycle = 0
    while cycle < max_cycle:
        phi_tot = phi_opt + bc
        drho_ions = get_drho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        jac_diag = inv_eps * drho_ions

        v = cupy.zeros(tot_ngrids, dtype=cupy.float64)
        b = cupy.negative(res_outer, out=b)

        res_inner[:] = b
        rnorm = fnorm
        inner_thresh = C_TST * fnorm**(1.0 + p_TST)
        inner = 0
        while inner < inner_max_cycle:
            if rnorm <= inner_thresh and rnorm < fnorm:
                break
            dv = precond(res_inner, dv)
            v -= dv
            Av = A(v, Av)
            dv = cupy.multiply(jac_diag, v, out=dv)
            res_inner = cupy.subtract(b, Av, out=res_inner)
            res_inner -= dv
            rnorm = cupy.linalg.norm(res_inner)
            inner += 1

        # Line search by backtracking.
        damping = 1.0
        accepted = False
        while damping > 1.0e-5:
            res_try, rho_ions_try = residual(phi_opt + damping * v, res_old)
            if res_try is not None:
                fnorm_try = cupy.linalg.norm(res_try)
                if fnorm_try < (1.0 - damping / 10**4) * fnorm:
                    accepted = True
                    break
            damping *= 0.5

        if not accepted:
            logger.warn(solvent_obj, 'Newton line search failed at cycle %d '
                        '(||F|| = %4.3e, inner = %d).', cycle + 1, fnorm, inner)
            res_try, rho_ions_try = residual(phi_opt + damping * v, res_old)
            if res_try is None:
                logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
                return None, None, None
            fnorm_try = cupy.linalg.norm(res_try)

        phi_opt = phi_opt + damping * v
        res_outer, rho_ions = res_try, rho_ions_try
        fnorm = fnorm_try
        res_new, res_old = res_old, res_new   # accepted trial becomes current
        cycle += 1
        logger.info(solvent_obj, 'PBE Iteration %3d ||F|| = %4.3e inner = %2d damp = %4.3e',
                    cycle, fnorm, inner, damping)

        if fnorm < 1.0e-9:
            t0 = logger.timer(solvent_obj, 'phi_tot', *t0)
            return finalize(phi_opt, rho_ions)

    raise RuntimeError('Newton PBE solver failed to converge.')


class NLPBE(pbe.NLPBE):
    """GPU-accelerated DINMH algorithm for solving NLPBE"""
    def __init__(self, mol, *args, **kwargs):
        self.handle = None
        super().__init__(mol, *args, **kwargs)

    def build(self):
        super().build()
        device_id = cupy.cuda.Device().id
        with cupy.cuda.Device(device_id):
            # Create cuSPARSE handle
            handle = ctypes.c_void_p(0)
            libamgcl._cusparseCreate(ctypes.byref(handle))
            self.handle = handle
        return self

    def _gen_get_rho_ions(self):
        return rho_ions_one_to_one

    def _gen_drho_ions(self):
        return drho_ions_one_to_one

    def _get_vmat(self, phi_pol):
        mol = self.mol
        coords = self.grids.coords
        spacing = self.grids.spacing
        nao = mol.nao
        tot_ngrids = self.grids.get_ngrids()

        nbatch = 256*256
        phi_pol = cupy.asarray(phi_pol)
        buf = cupy.empty((nbatch, nao), order='C')
        vmat = cupy.zeros((nao, nao), order='C')
        verbose = mol.verbose
        mol.verbose = 0
        log = gpulogger.new_logger(self, verbose)
        t0 = log.init_timer()
        for p0, p1 in lib.prange(0, tot_ngrids, nbatch):
            ao = _ni.eval_ao(mol, coords[p0:p1])
            buf[:p1-p0] = ao * phi_pol[p0:p1, None]
            vmat -= cupy.dot(buf[:p1-p0].T, ao)
        vmat *= spacing**3
        mol.verbose = verbose
        t0 = log.timer('v_diel', *t0)
        return vmat.get()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        if self.handle is not None:
            _release_caches(self)
            self.hierarchy = None
            self.operator = None
            self.precond = None
            self.handle = None
        return self

    gradient = gradient
    make_phi_sol = make_phi_sol
    make_operator = make_operator
    make_precond = make_precond
    make_phi = make_phi