import numpy
import cupy
import scipy
import ctypes
from pyscf import df
from pyscf import gto
from pyscf import lib
from pyscf.scf import _vhf
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import cart2sph_by_l
from gpu4pyscf.scf.int4c2e import libgint
from gpu4pyscf.df.int3c2e import make_fake_mol, get_pairing, get_ao_pairs
from gpu4pyscf.df.int3c2e import VHFOpt
from gpu4pyscf.df import int3c2e_bdiv
from cupyx.scipy.linalg import solve_triangular

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

    if dm is None:
        dm = solvent_obj._dm
    if coords is None:
        coords = solvent_obj.grids.coords

    verbose = solvent_obj.verbose
    log = logger.new_logger(solvent_obj, verbose)
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

    Vele = cupy.empty(tot_ngrids)
    for p0, p1 in lib.prange(0, tot_ngrids, nbatch):
        int2c = int2c2e_grids(auxmol, coords[p0:p1], direct_scf_tol=1e-14)
        Vele[p0:p1] = int2c.dot(d)
        int2c = None
    Vele = Vele.get()

    MEP = Vnuc - Vele
    t0 = log.timer('phi_sol', *t0)
    return lib.tag_array(MEP, Vnuc=Vnuc, Vele=-Vele)