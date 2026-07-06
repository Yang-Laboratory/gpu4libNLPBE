#include <cusparse_v2.h>
#include <vector>
#include <amgcl/util.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: hand freed heap back to the OS
#endif

static inline void release_to_os()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

typedef amgcl::backend::cuda<double> Backend;

template <template <class> class Relax>
using AMG = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, Relax>;

typedef AMG<amgcl::relaxation::spai0>        AMG_spai0;

struct Hierarchy {
    // virtual void rebuild(double* data) = 0;   // val has nnz entries
    virtual void vcycle(double* rhs, double* x) = 0;
    virtual ~Hierarchy() {}
};

template <class Amg>
struct HierarchyImpl : Hierarchy {
    int tot_ngrids;
    std::vector<int> indptr;
    std::vector<int> indices;
    thrust::device_vector<double> d_rhs;
    thrust::device_vector<double> d_x;
    Amg amg;

    HierarchyImpl(int tot_ngrids_,
                  int* indptr_,
                  int *indices_,
                  std::vector<double> data,
                  const typename Amg::params& prm,
                  const typename Amg::backend_params& bprm) :
                  tot_ngrids(tot_ngrids_),
                  indptr(indptr_, indptr_ + tot_ngrids_ + 1),
                  indices(indices_, indices_ + indptr_[tot_ngrids_]),
                  amg(std::tie(tot_ngrids, indptr, indices, data), prm, bprm),
                  d_rhs(tot_ngrids_),
                  d_x(tot_ngrids_)
    {
        indptr.shrink_to_fit();
        indices.shrink_to_fit();
    }

    void vcycle(double *rhs, double *x) override {
        // Load vectors to the device
        thrust::device_ptr<double> p_rhs(rhs);
        thrust::device_ptr<double> p_x(x);
        thrust::copy(p_rhs, p_rhs + tot_ngrids, d_rhs.begin());
        amg.apply(d_rhs, d_x);
        thrust::copy(d_x.begin(), d_x.end(), p_x);
    }
};

template <class Amg>
static void set_params(typename Amg::params& prm, int max_levels,
                       int coarse_enough, int npre, int npost, int ncycle,
                       int pre_cycles) {
    prm.direct_coarse = false;                 // coarsest level: direct solve
    prm.allow_rebuild = true;
    if (max_levels > 0)    prm.max_levels    = static_cast<unsigned>(max_levels);
    if (coarse_enough > 0) prm.coarse_enough = static_cast<unsigned>(coarse_enough);
    if (npre  >= 0)        prm.npre   = static_cast<unsigned>(npre);
    if (npost >= 0)        prm.npost  = static_cast<unsigned>(npost);
    if (ncycle > 0)        prm.ncycle = static_cast<unsigned>(ncycle);
    if (pre_cycles >= 0)   prm.pre_cycles = static_cast<unsigned>(pre_cycles);
}

template <class Amg>
static Hierarchy* build(cusparseHandle_t handle, int tot_ngrids,
                        int *indptr, int* indices, double *data,
                        int max_levels, int coarse_enough, int npre, int npost,
                        int ncycle, int pre_cycles) {
    typename Amg::params prm;
    set_params<Amg>(prm, max_levels, coarse_enough, npre, npost, ncycle, pre_cycles);
    // backend params
    Backend::params bprm;
    bprm.cusparse_handle = handle;

    std::vector<double> v(data, data + indptr[tot_ngrids]);
    return new HierarchyImpl<Amg>(tot_ngrids, indptr, indices, std::move(v), prm, bprm);
}

extern "C" {
    void _cusparseCreate(cusparseHandle_t *handle) {
        cusparseCreate(handle);
    }
    
    void _cusparseDestroy(cusparseHandle_t handle) {
        cusparseDestroy(handle);
    }
    // Create amg hirarchy
    void* amg_create(cusparseHandle_t handle, int tot_ngrids, int *indptr, int *indices,
                    double *data, int max_levels, int coarse_enough, int relax, int npre,
                    int npost, int ncycle, int pre_cycles) {
        Hierarchy* h;
        h = build<AMG_spai0>(handle, tot_ngrids, indptr, indices, data, max_levels,
                            coarse_enough, npre, npost, ncycle, pre_cycles);
        return static_cast<void*>(h);
    }

    // V-cycle
    void amg_vcycle(void* h, double *rhs, double *x, int tot_ngrids) {
        (void)tot_ngrids;
        static_cast<Hierarchy*>(h)->vcycle(rhs, x);
    }

    // Destroy hirarchy and retrive memory.
    void amg_destroy(void* h) {
        delete static_cast<Hierarchy*>(h);
        release_to_os();
    }

    // Av = out
    void csr_matvec(cusparseHandle_t handle, int *indptr, int *indices, double *data, double *v, double *out, int tot_ngrids, int nnz) {
        double alpha = 1.0;
        double beta  = 0.0;
        void *dBuffer = NULL;
        size_t bufferSize = 0;

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateCsr(&matA, tot_ngrids, tot_ngrids, nnz, indptr, indices, data,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateDnVec(&vecX, tot_ngrids,   v, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, tot_ngrids, out, CUDA_R_64F);

        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

        cudaMalloc(&dBuffer, bufferSize);
        cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cudaFree(&dBuffer);
    }
}