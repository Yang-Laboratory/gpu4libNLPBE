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

from pyscf import df
from pyscf import gto
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import pack_tril
from gpu4pyscf.gto.mole import cart2sph_by_l
from gpu4pyscf.scf.int4c2e import libgint
from gpu4pyscf.df.int3c2e import make_fake_mol, get_pairing, get_ao_pairs
from gpu4pyscf.df.int3c2e import VHFOpt
from gpu4pyscf.df import int3c2e_bdiv
from cupyx.scipy.linalg import solve_triangular


PI = numpy.pi
KB2HARTREE = BOLTZMANN / HARTREE2J
M2HARTREE = AVOGADRO*BOHR**3*1.e-27

# libamgcl = lib.load_library(os.path.join(monkey.__path__[0], 'lib', 'libdinmh.so'))
libamgcl = lib.load_library(os.path.join(os.getcwd(), 'lib', 'libdinmh.so'))
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

def fakemol_for_charges(coords, expnt=1e16):
    """Generate Dirac-delta function centered at cubic grids.
    Unlike PySCF, this function places two pGTOs at each grids.
    The resulting fakemol will be separable from auxmol (#pGTO=1).
    C*exp(-ar**2) -> 0.5C*exp(-ar**2) + 0.5C*exp(-ar**2)

    Args:
        coords (_type_): _description_
        expnt (_type_, optional): _description_. Defaults to 1e16.
    """
    fakemol = gto.fakemol_for_charges(coords, expnt=expnt)
    env0 = fakemol._env
    bas0 = fakemol._bas
    expt = env0[bas0[0, gto.PTR_EXP  ]] # Same exponenet
    cont = env0[bas0[0, gto.PTR_COEFF]] * 0.5 # Halve contraction coeff
    split = numpy.array([expt, expt, cont, cont], dtype=numpy.float64)
    fakemol._env = numpy.concatenate([env0, split])

    ptr_env = len(env0)
    bas = bas0.copy()
    bas[:, gto.NPRIM_OF ] = 2           # Two pGTOs
    bas[:, gto.PTR_EXP  ] = ptr_env     # Where to find the exponent
    bas[:, gto.PTR_COEFF] = ptr_env + 2 # Where to find the contraction coeff
    fakemol._bas = bas
    fakemol._built = True
    return fakemol

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

#!!!!!!!!!!!!!!
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

def _build(self, cutoff):
    """Custom VHFOpt.build for (P|delta) at cubic grids.
    Note that GPU4PySCF reorders _bas wrt orbital angular momentum.

    Args:
        cutoff (_type_): _description_
    """
    # _mol: auxmol, _auxmol: fakemol
    _mol = self.mol
    _auxmol = self.auxmol

    # mol: auxmol, auxmol: fakemol
    mol = _basis_seg_contraction(_mol, allow_replica=True, sparse_coeff=True)[0]
    auxmol = _basis_seg_contraction(_auxmol, allow_replica=True, sparse_coeff=True)[0]

    _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = _sort_mol(mol)

    self.nctr = len(uniq_l_ctr)
    self.l_ctr_counts = l_ctr_counts

    # sort fake mol
    fake_mol = make_fake_mol()
    _, _, fake_uniq_l_ctr, fake_l_ctr_counts = _sort_mol(fake_mol)

    # sort auxiliary mol
    _sorted_auxmol, sorted_aux_idx, aux_uniq_l_ctr, aux_l_ctr_counts = _sort_mol(auxmol)
    self.aux_l_ctr_counts = aux_l_ctr_counts

    _tot_mol = _sorted_mol + fake_mol + _sorted_auxmol
    _tot_mol.cart = True

    # shift atom indices back to actual atom indices
    nbas = _sorted_mol.nbas + 1 # auxmol start index
    _tot_mol._bas[nbas:, gto.ATOM_OF] -= (mol.natm+1) 
    self._tot_mol = _tot_mol

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

    # pairing auxiliary basis with fake basis set
    fake_l_ctr_offsets = numpy.append(0, numpy.cumsum(fake_l_ctr_counts))
    fake_l_ctr_offsets += l_ctr_offsets[-1]
    aux_l_ctr_offsets = numpy.append(0, numpy.cumsum(aux_l_ctr_counts))

    # contraction coefficient for auxiliary basis
    cart_aux_loc = _sorted_auxmol.ao_loc_nr(cart=True)
    sph_aux_loc = _sorted_auxmol.ao_loc_nr(cart=False)
    self.cart_aux_loc = [cart_aux_loc[cp] for cp in aux_l_ctr_offsets]
    self.sph_aux_loc = [sph_aux_loc[cp] for cp in aux_l_ctr_offsets]
    self.aux_angular = [l[0] for l in aux_uniq_l_ctr]

    self._aux_ao_idx = aux_ao_idx(_auxmol, sorted_aux_idx).astype(numpy.int32)

    ao_loc = _sorted_mol.ao_loc_nr(cart=_mol.cart)
    self.ao_pairs_row, self.ao_pairs_col = get_ao_pairs(pair2bra, pair2ket, ao_loc)
    cderi_row = numpy.hstack(self.ao_pairs_row)
    cderi_col = numpy.hstack(self.ao_pairs_col)
    self.cderi_row = cderi_row
    self.cderi_col = cderi_col
    self.cderi_diag = numpy.argwhere(cderi_row == cderi_col)[:,0]

    aux_pair2bra = []
    aux_pair2ket = []
    aux_log_qs = []
    aux_l_ctr_offsets += fake_l_ctr_offsets[-1]
    for p0, p1 in zip(aux_l_ctr_offsets[:-1], aux_l_ctr_offsets[1:]):
        aux_pair2bra.append(numpy.arange(p0,p1,dtype=numpy.int32))
        aux_pair2ket.append(fake_l_ctr_offsets[0] * numpy.ones(p1-p0, dtype=numpy.int32))
        aux_log_qs.append(numpy.ones(p1-p0, dtype=numpy.int32))

    self.aux_log_qs = aux_log_qs.copy()
    pair2bra += aux_pair2bra
    pair2ket += aux_pair2ket

    self.aux_pair2bra = aux_pair2bra
    self.aux_pair2ket = aux_pair2ket

    self.pair2bra = pair2bra
    self.pair2ket = pair2ket
    self.l_ctr_offsets = l_ctr_offsets

    self._bpcache = {}

    bas_pairs_locs = numpy.append(0, numpy.cumsum([x.size for x in pair2bra])).astype(numpy.int32)
    self.bas_pairs_locs = bas_pairs_locs
    ncptype = len(self.log_qs)
    self.aosym = True
    self.cp_idx, self.cp_jdx = numpy.tril_indices(ncptype)

    if _mol.cart:
        self.ao_loc = self.cart_ao_loc
    else:
        self.ao_loc = self.sph_ao_loc
    if _auxmol.cart:
        self.aux_ao_loc = self.cart_aux_loc
    else:
        self.aux_ao_loc = self.sph_aux_loc

    self._sorted_mol = _sorted_mol
    self._sorted_auxmol = _sorted_auxmol


def int2c2e_grids(auxmol, coords, direct_scf_tol=1e-14, expnt=1e16):
    """Computes (P|delta) implemented by int2e routine in GPU4PySCF.

    Args:
        auxmol (_type_): _description_
        coords (_type_): _description_
        expnt (_type_, optional): _description_. Defaults to 1e16.
    """
    naux = auxmol.nao
    tot_ngrids = coords.shape[0]
    fakemol = fakemol_for_charges(coords, expnt=expnt)

    aux_fakemol = gto.conc_mol(auxmol, fakemol)
    aux_fakemol.cart = auxmol.cart

    intopt = VHFOpt(auxmol, aux_fakemol, 'int2e')
    _build(intopt, direct_scf_tol)

    # _tot_mol = _sorted_mol + fake_mol + _sorted_auxmol
    # fake_mol = normalized Gaussian with zero exponent
    # _tot_mol = sorted auxmol + fake_mol + sorted aux_fakemol
    # ATOM_OF=0: atom index, ANG_OF=1: angular momentum
    # NPRIM_OF=2: # of pGTOs, NCTR_OF=3: # of cGTOs
    # shift = auxmol.natm + 1 # Pointing the fake_mol start idx
    nbas = intopt._sorted_mol.nbas + 1 # Pointing the fake_mol nbas start idx
    mask = intopt._sorted_auxmol._bas[:, gto.NPRIM_OF] == 2 # fakemol indices
    intopt._tot_mol._bas[nbas:][mask, gto.ATOM_OF] += auxmol.natm + 1

    nao_cart = intopt._sorted_mol.nao
    naux_cart = intopt._sorted_auxmol.nao
    norb_cart = nao_cart + naux_cart + 1

    aux_l_ctr_offsets = numpy.append(0, numpy.cumsum(intopt.aux_l_ctr_counts))
    aux_bas_sorted = intopt._sorted_auxmol._bas
    cart_aux_loc = numpy.asarray(intopt.cart_aux_loc, dtype=numpy.int32)
    sph_aux_loc = numpy.asarray(intopt.sph_aux_loc, dtype=numpy.int64)
    aux_ao_idx = numpy.asarray(intopt._aux_ao_idx, dtype=numpy.int64)

    aux_ids = []
    fake_ids = []
    for cp_id in range(len(intopt.aux_log_qs)):
        p0 = int(aux_l_ctr_offsets[cp_id])
        if int(aux_bas_sorted[p0, gto.NPRIM_OF]) == 2:
            fake_ids.append(cp_id)
        else:
            aux_ids.append(cp_id)

    int2c = cupy.zeros((naux, tot_ngrids), dtype=cupy.float64, order='F')
    stream = cupy.cuda.get_current_stream()
    log_cutoff = numpy.log(direct_scf_tol)
    omega = 0.0

    max_nk = max(cart_aux_loc[k_id+1] - cart_aux_loc[k_id] for k_id in aux_ids)
    max_nl = max(cart_aux_loc[l_id+1] - cart_aux_loc[l_id] for l_id in fake_ids)
    
    buf = cupy.empty(max_nk*max_nl, dtype=cupy.float64)

    for k_id in aux_ids:
        lk = intopt.aux_angular[k_id]
        k0, k1 = cart_aux_loc[k_id], cart_aux_loc[k_id+1]
        nk = k1 - k0

        aux_idx = aux_ao_idx[sph_aux_loc[k_id]:sph_aux_loc[k_id+1]]
        aux_idx = cupy.asarray(aux_idx)

        log_q_k = intopt.aux_log_qs[k_id]
        bins_locs_k = numpy.array([0, len(log_q_k)], dtype=numpy.int32)
        bins_floor_k = numpy.array([100], dtype=numpy.double)
        cp_k_id = k_id + len(intopt.log_qs)

        for l_id in fake_ids:
            l0, l1 = cart_aux_loc[l_id], cart_aux_loc[l_id+1]
            nl = l1 - l0

            log_q_l = intopt.aux_log_qs[l_id]
            bins_locs_l = numpy.array([0, len(log_q_l)], dtype=numpy.int32)
            bins_floor_l = numpy.array([100], dtype=numpy.double)
            cp_l_id = l_id + len(intopt.log_qs)

            int2c_blk = buf[:nk*nl].reshape((nk, nl), order='F')
            int2c_blk.fill(0.0)
            ao_offsets = numpy.array([nao_cart + 1 + k0, nao_cart,
                                      nao_cart + 1 + l0, nao_cart], dtype=numpy.int32)
            strides = numpy.array([1, nao_cart, nk, nao_cart*nao_cart], dtype=numpy.int32)
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
                t_cart = cupy.ascontiguousarray(int2c_blk)
                t_cart = t_cart.reshape(nshl, c2s.shape[0], nl)
                int2c_blk = cupy.einsum('min,ip->mpn', t_cart, c2s, optimize=True)
                int2c_blk = int2c_blk.reshape(nshl * c2s.shape[1], nl)

            grid_idx = aux_ao_idx[sph_aux_loc[l_id]:sph_aux_loc[l_id + 1]] - naux
            grid_idx = cupy.asarray(grid_idx)
            
            int2c[aux_idx[:,None], grid_idx[None,:]] = int2c_blk

    stream.synchronize()
    return int2c.T

def make_phi_sol(solvent_obj, dm=None, coords=None):
    """GPU-accelerated solute electrostati potential based on density fitting.

    Args:
        solvent_obj (_type_): _description_
        dm (_type_, optional): _description_. Defaults to None.
        coords (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    t0 = log.init_timer()
    if dm is None: dm = solvent_obj._dm
    if coords is None: coords = solvent_obj.grids.coords
    _intermediates = solvent_obj._intermediates

    verbose = solvent_obj.verbose
    log = logger.new_logger(solvent_obj, verbose)

    mol = solvent_obj.mol
    tot_ngrids = coords.shape[0]

    # Nuclear part
    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    dist[dist < 1.0e-100] = numpy.inf
    Vnuc = numpy.tensordot(1.0e0 / dist, Z, axes=([0], [0]))

    # Electronic part
    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    auxmol = _intermediates['auxmol']
    erifile = _intermediates['erifile']
    nao = mol.nao
    naux = auxmol.nao

    dms = numpy.asarray(dm.real)
    dm_tril = pack_tril(dms + dms.T)
    idx = numpy.arange(nao)
    idx = idx * (idx + 1) // 2 + idx
    dm_tril[idx] *= 0.5

    with h5py.File(erifile, 'r') as feri:
        int2c2e = cupy.asarray(feri['int2c2e'])
        int3c2e = cupy.asarray(feri['int3c2e'])

    g = dm_tril.dot(int3c2e)
    
    nbatch = 256*256
    # j2c = auxmol.intor('int2c2e')
    # j2c = cupy.asarray(j2c)    
    # rhoj = int3c2e_bdiv.contract_int3c2e_dm(mol, auxmol, dms)
    cd_low = cupy.linalg.cholesky(int2c2e)
    y = solve_triangular(cd_low, g, lower=True, overwrite_b=True)
    cK = solve_triangular(cd_low, y, trans='T', lower=True, overwrite_b=True)

    Vele = cupy.empty(tot_ngrids)
    for p0, p1 in lib.prange(0, tot_ngrids, nbatch):
        ints = int2c2e_grids(auxmol, coords[p0:p1], direct_scf_tol=1e-14)
        Vele[p0:p1] = ints.dot(cK)
        ints = None
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
    solver = solvent_obj.solver
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    if grad_lneps is None:
        _intermediates = solvent_obj._intermediates
        eps = _intermediates['eps']
        grad_eps = _intermediates['grad_eps']
        grad_lneps = grad_eps / eps[:,None]

    L = solver.L # nabla**2 = -L / spacing**2
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
    solver = solvent_obj.solver
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    L = solver.L
    precond = L / spacing**2 + scipy.sparse.diags(drho_ions_scr.get(), format='csr')

    if solvent_obj.hirarchy is None:
        verbose = solvent_obj.verbose
        log = gpulogger.new_logger(solvent_obj, verbose)
        t0 = log.init_timer()
        handle = solvent_obj.handle
        hirarchy = libamgcl.amg_create(handle,
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
        solvent_obj.hirarchy = hirarchy
        t0 = log.timer('make_precond', *t0)
    hirarchy = solvent_obj.hirarchy

    def vcycle(r, out):
        libamgcl.amg_vcycle(ctypes.c_void_p(hirarchy),
                            r.data.ptr,
                            out.data.ptr,
                            ctypes.c_int(tot_ngrids))
        return out

    return vcycle

def _release_caches(solvent_obj):
    """Release memory after make_phi"""
    if getattr(solvent_obj, 'hirarchy', None) is not None:
        hirarchy = solvent_obj.hirarchy
        libamgcl.amg_destroy(ctypes.c_void_p(hirarchy))
    libamgcl._cusparseDestroy(solvent_obj.handle)
    solvent_obj.hirarchy = None

def make_phi(solvent_obj, bias=None, phi_sol=None, rho_sol=None):
    if solvent_obj._intermediates is None: solvent_obj.build()
    _intermediates = solvent_obj._intermediates

    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = solvent_obj.grids.get_ngrids()
    T = solvent_obj.T
    spacing = solvent_obj.grids.spacing
    stern_sam = solvent_obj.stern_sam / BOHR
    cb = solvent_obj.cb * M2HARTREE
    pzc = solvent_obj.pzc / HARTREE2EV
    ref_pot = solvent_obj.ref_pot / HARTREE2EV
    jump_coeff = solvent_obj.jump_coeff

    eps = _intermediates['eps']
    lambda_r = _intermediates['lambda_r']
    grad_eps = _intermediates['grad_eps']

    solver = solvent_obj.solver

    max_cycle = solvent_obj.max_cycle # Newton cycle
    inner_max_cycle = 100 # Inner cycle

    C_TST = 0.01
    p_TST = 1.0

    grad_lneps = grad_eps / eps[:, None]
    get_rho_ions = solvent_obj._gen_get_rho_ions()
    get_drho_ions = solvent_obj._gen_drho_ions()

    inv_eps = cupy.array(4.0 * PI / eps)

    A = solvent_obj.make_operator(grad_lneps)

    if cb == 0.0:
        drho_ions_scr = None
    else:
        drho_ions_scr = -inv_eps * get_drho_ions(solvent_obj, cupy.zeros(tot_ngrids), cb, lambda_r, T)

    precond = solvent_obj.make_precond(drho_ions_scr)
    rho_sol = cupy.asarray(rho_sol)
    grad_lneps = cupy.asarray(grad_lneps)
    eps = cupy.asarray(eps)

    def residual(phi_tot, out):
        rho_ions = get_rho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        if cupy.isnan(rho_ions).any():
            return None, rho_ions
        rho_tot = rho_sol + rho_ions
        A(phi_tot, out)
        out += inv_eps * rho_tot
        return out, rho_ions

    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info(solvent_obj, ' |  Poisson-Boltzmann Solver with the Multigrid Scheme  |')
    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

    phi_tot = cupy.zeros(tot_ngrids, dtype=cupy.float64)
    res_old = cupy.empty(tot_ngrids, dtype=cupy.float64)
    res_new = cupy.empty(tot_ngrids, dtype=cupy.float64)
    res_outer, rho_ions = residual(phi_tot, res_new)

    if res_outer is None:
        logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
        return None, None
    
    fnorm = cupy.linalg.norm(res_outer)

    res_inner = cupy.empty(tot_ngrids, dtype=cupy.float64)
    dv = cupy.empty(tot_ngrids, dtype=cupy.float64)
    Av = cupy.empty(tot_ngrids, dtype=cupy.float64)
    b = cupy.empty(tot_ngrids, dtype=cupy.float64)

    t0 = (logger.process_clock(), logger.perf_counter())

    cycle = 0
    while cycle < max_cycle:
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
            res_try, rho_ions_try = residual(phi_tot + damping * v, res_old)
            if res_try is not None:
                fnorm_try = cupy.linalg.norm(res_try)
                if fnorm_try < (1.0 - damping / 10**4) * fnorm:
                    accepted = True
                    break
            damping *= 0.5

        if not accepted:
            logger.warn(solvent_obj, 'Newton line search failed at cycle %d '
                        '(||F|| = %4.3e, inner = %d).', cycle + 1, fnorm, inner)
            res_try, rho_ions_try = residual(phi_tot + damping * v, res_old)
            if res_try is None:
                logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
                return None, None
            fnorm_try = numpy.linalg.norm(res_try)

        phi_tot = phi_tot + damping * v
        res_outer, rho_ions = res_try, rho_ions_try
        fnorm = fnorm_try
        res_new, res_old = res_old, res_new   # accepted trial becomes current
        cycle += 1
        logger.info(solvent_obj, 'PBE Iteration %3d ||F|| = %4.3e inner = %2d damp = %4.3e',
                    cycle, fnorm, inner, damping)

        if fnorm < 1.0e-9:
            t0 = logger.timer(solvent_obj, 'phi_tot', *t0)
            return phi_tot.get(), rho_ions.get()

    logger.info(solvent_obj, 'Newton PBE failed to converge.')
    raise RuntimeError('Newton PBE solver failed to converge.')


class NLPBE(pbe.NLPBE):
    """GPU-accelerated DINMH algorithm for solving NLPBE"""
    def __init__(self, mol, *args, **kwargs):
        self.handle = None
        super().__init__(mol, *args, **kwargs)

    def build(self):
        super().build()
        # Create cuSPARSE handle
        handle = ctypes.c_void_p(0)
        libamgcl._cusparseCreate(ctypes.byref(handle))
        self.handle = handle
        return self

    def _gen_get_rho_ions(self):
        return rho_ions_one_to_one

    def _gen_drho_ions(self):
        return drho_ions_one_to_one

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        if self.handle is not None:
            _release_caches(self)
            self.hirarchy = None
            self.operator = None
            self.precond = None
            self.handle = None
        return self

    make_operator = make_operator
    make_precond = make_precond
    make_phi = make_phi