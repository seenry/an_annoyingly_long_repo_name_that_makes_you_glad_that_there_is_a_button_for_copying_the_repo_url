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

    // Datatype Handling
    MPI_Aint true_extent, true_lb, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPI_Aint necessary_extent = (true_extent > extent) ? true_extent : extent;

    MPIR_CHKLMEM_DECL();
    if (comm_size == 1) {
        if (sendbuf != MPI_IN_PLACE) {
            mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
        goto fn_exit;
    }

    int depth = 1;
    while (0x1 << depth < comm_size) depth++;

    int* skips;
    int* block_counts; int* block_starts;
    void* partial_buf; void* tmp_recv_buf;
    MPIR_CHKLMEM_MALLOC(skips, (depth+1) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(block_counts, (comm_size * 2) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(block_starts, (comm_size * 2) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(partial_buf, count * necessary_extent);
    MPIR_CHKLMEM_MALLOC(tmp_recv_buf, (comm_size / 2) * necessary_extent);

    partial_buf = (void*) ((char*) partial_buf - true_lb);
    tmp_recv_buf = (void*) ((char*) tmp_recv_buf - true_lb);

    skips[depth] = comm_size;
    for (int i = depth-1; i >= 0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }

    // Might be worth thinking about different partitioning schemes
    block_counts[0] = (count / comm_size)
                    + ((count % comm_size) ? 1 : 0);
    block_starts[0] = 0;
    for (int i = 1; i < (comm_size * 2); i++) {
        int j = i % comm_size;
        block_counts[i] = (count / comm_size)
                        + ((j < count % comm_size) ? 1 : 0);
        block_starts[i] = block_starts[i-1] + block_counts[i-1];
    }

    // Initialize rotated buffer
    for (int i = 0; i < comm_size; i++) {
        int rel = rank + i;
        int bcount = block_counts[rel];
        if (bcount != 0) {
            int src_offset = block_starts[rel % comm_size];
            int dst_offset = block_starts[rel] - block_starts[rank];
            MPIR_Localcopy((char*)sendbuf + (src_offset * extent), bcount, datatype,
                           (char*)partial_buf + (dst_offset * extent), bcount, datatype);
        }
    }

    MPIR_Request* requests[2];

    // Reduction
    for (int k = depth - 1; k >= 0; k--) {
        int to = (rank + skips[k]) % comm_size;
        int from = (rank - skips[k] + comm_size) % comm_size;

        int n_blocks = skips[k+1] - skips[k];

        int n_send = block_starts[rank + skips[k] + n_blocks] - block_starts[rank + skips[k]];
        if (n_send) {
            int send_offset = block_starts[rank + skips[k]] - block_starts[rank];
            mpi_errno = MPIC_Isend((char*)partial_buf + (send_offset * extent), n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[0], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        }
        int n_recv = block_starts[rank + n_blocks] - block_starts[rank];
        if (n_recv) {
            mpi_errno = MPIC_Irecv(tmp_recv_buf, n_recv, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[(!!n_send)]);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (n_send || n_recv) {
            mpi_errno = MPIC_Waitall((!!n_send) + (!!n_recv), requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (n_recv) {
            mpi_errno = MPIR_Reduce_local(tmp_recv_buf, partial_buf, n_recv, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    // Scatter
    for (int k = 0; k < q; k++) {
        int to = (rank - skips[k] + comm_size) % comm_size;
        int from = (rank + skips[k]) % comm_size;
        
        int n_blocks = skips[k+1] - skips[k];

        int n_send = block_starts[rank + n_blocks] - block_starts[rank];
        if (n_send) {
            mpi_errno = MPIC_Isend(partial_buf, n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[0], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        }
        int n_recv = block_starts[rank + skips[k] + n_blocks] - block_starts[rank + skips[k]];
        if (n_recv) {
            int recv_offset = block_starts[rank + skips[k]] - block_starts[rank];
            mpi_errno = MPIC_Irecv((char*)partial_buf + (recv_offset * extent), n_recv, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[(!!n_send)]);
        }
        if (n_send || n_recv) {
            mpi_errno = MPIC_Waitall((!!n_send) + (!!n_recv), requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    // Undo rotation
    for (int i = 0; i < comm_size; i++) {
        int rel = rank + i;
        int bcount = block_counts[rel];
        if (bcount != 0) {
            int src_offset = block_starts[rel] - block_starts[rank];
            int dst_offset = block_starts[rel % comm_size];
            MPIR_Localcopy((char*)partial_buf + (src_offset * extent), bcount, datatype,
                           (char*)recvbuf + (dst_offset * extent), bcount, datatype);
        }
    }

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
