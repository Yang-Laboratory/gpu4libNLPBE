import numpy
import cupy
import cupyx
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