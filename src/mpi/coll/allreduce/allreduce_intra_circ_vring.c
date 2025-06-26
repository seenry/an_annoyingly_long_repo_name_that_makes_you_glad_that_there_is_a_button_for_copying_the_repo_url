#include "mpiimpl.h"
#include <math.h>

int MPIR_Allreduce_intra_circ_vring(const void* sendbuf,
                                    void* recvbuf,
                                    MPI_Aint count,
                                    MPI_Datatype datatype,
                                    MPI_Op op,
                                    MPIR_Comm* comm,
                                    MPIR_Errflag_t errflag)
{
    int mpi_errno = MPI_SUCCESS;

    int comm_size, rank;
    comm_size = comm->local_size;
    rank = comm->rank;
    
    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
        goto fn_exit;
    }

    MPIR_CHKLMEM_DECL();

    // I just yoinked this so I hope it does what I expect
    MPI_Aint true_extent, true_lb, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPI_Aint necessary_extent = (true_extent > extent) ? true_extent : extent;

    void* tmp_buf;
    MPIR_CHKLMEM_MALLOC(tmp_buf, count * necessary_extent);
    tmp_buf = (void*) ((char*) tmp_buf - true_lb);
    
    void* modified_send_buf;
    MPIR_CHKLMEM_MALLOC(modified_send_buf, count * necessary_extent);
    modified_send_buf = (void*) ((char*) modified_send_buf - true_lb);

    void* true_send = (void*) (uint64_t) sendbuf;
    if (sendbuf == MPI_IN_PLACE) {
        MPIR_CHKLMEM_MALLOC(true_send, count * necessary_extent);
        mpi_errno = MPIR_Localcopy(recvbuf, count, datatype, true_send, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    int depth = 1;
    while (0x1<<depth < comm_size) {
        depth++;
    }
    int* skips;
    MPIR_CHKLMEM_MALLOC(skips, (depth+1) * sizeof(int));
    skips[depth] = comm_size;
    for (int i = depth-1; i>=0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }
    
    MPIR_Request* requests[2];

    {
        int to = (rank - 1 + comm_size) % comm_size;
        int from = (rank + 1) % comm_size;
        mpi_errno = MPIC_Irecv(recvbuf, count, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[0]);
        MPIR_ERR_CHECK(mpi_errno);
        mpi_errno = MPIC_Isend(true_send, count, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[1], errflag);
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = MPIC_Waitall(2, requests, MPI_STATUSES_IGNORE);
        MPIR_ERR_CHECK(mpi_errno);
    }
    
    for (int k = 1; k < depth; k++) {
        int adjust = skips[k+1] & 0x1;
        int to = (rank - (skips[k] - adjust) + comm_size) % comm_size;
        int from = (rank + (skips[k] - adjust)) % comm_size;
        
        mpi_errno = MPIC_Irecv(tmp_buf, count, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[0]);
        MPIR_ERR_CHECK(mpi_errno);
        if (adjust) {
            mpi_errno = MPIC_Isend(recvbuf, count, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[1], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            mpi_errno = MPIR_Localcopy(true_send, count, datatype, modified_send_buf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIR_Reduce_local(recvbuf, modified_send_buf, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIC_Isend(modified_send_buf, count, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[1], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        }
        mpi_errno = MPIC_Waitall(2, requests, MPI_STATUSES_IGNORE);
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
        MPIR_ERR_CHECK(mpi_errno);
    }
    
    mpi_errno = MPIR_Reduce_local(true_send, recvbuf, count, datatype, op);
    MPIR_ERR_CHECK(mpi_errno);

    MPIR_CHKLMEM_FREEALL();
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
