#include "mpiimpl.h"
#include <math.h>
#include <unistd.h>

int MPIR_Allreduce_intra_circ_rs_ag(const void* sendbuf,
                                       void* recvbuf,
                                       MPI_Aint count,
                                       MPI_Datatype datatype,
                                       MPI_Op op,
                                       MPIR_Comm* comm,
                                       MPIR_Errflag_t errflag)
{
    int mpi_errno = MPI_SUCCESS;
    
    MPIR_Assert(MPIR_Op_is_commutative(op));

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

    // Type size logic
    MPI_Aint true_extent, true_lb, extent;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPI_Aint necessary_extent = (true_extent > extent) ? true_extent : extent;
    
    // Get CGA tree depth
    int cga_tree_depth = 1;
    while ((1 << cga_tree_depth) < comm_size) cga_tree_depth++;

    MPIR_CHKLMEM_DECL();

    // Generate skip list
    int* skips;
    MPIR_CHKLMEM_MALLOC(skips, (cga_tree_depth+1) * sizeof(int));
    skips[cga_tree_depth] = comm_size;
    for (int i = cga_tree_depth-1; i >= 0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }

    // Split input and calculate block offsets 
    int* offsets;
    MPIR_CHKLMEM_MALLOC(offsets, (comm_size * 2) * sizeof(int));
    offsets[0] = 0;
    for (int i = 1; i < comm_size * 2; i++) {
        int i_ = (i - 1 + comm_size) % comm_size;
        int prev_block = (count/comm_size)
                       + (i_ < (count%comm_size));

        offsets[i] = offsets[i-1] + prev_block;
    }

    // Initialize scratch buffers
    void* partial_buf__R; void* inter_buf__T;
    MPIR_CHKLMEM_MALLOC(partial_buf__R, count * necessary_extent);
    MPIR_CHKLMEM_MALLOC(inter_buf__T, count * necessary_extent); // possible to lower footprint
    partial_buf__R = (void*) ((char*) partial_buf__R - true_lb);
    inter_buf__T = (void*) ((char*) inter_buf__T - true_lb);

    const void* input_buffer = (sendbuf == MPI_IN_PLACE) ? recvbuf :sendbuf;

    // Initialize rotated buffer
    for (int i = 0; i < comm_size; i++) {
        int rel = rank + i;
        int b_count = offsets[rel+1] - offsets[rel];
        if (b_count != 0) {
            int src_offset = offsets[rel % comm_size];
            int dst_offset = offsets[rel] - offsets[rank];
            MPIR_Localcopy(((char*) input_buffer) + (src_offset * extent), b_count, datatype,
                           ((char*) partial_buf__R) + (dst_offset * extent), b_count, datatype);
        }
    }

    MPIR_Request* requests[2];

    // Reduction
    for (int k = cga_tree_depth - 1; k >= 0; k--) {
        int to = (rank + skips[k]) % comm_size;
        int from = (rank - skips[k] + comm_size) % comm_size;

        int n_blocks = skips[k+1] - skips[k];

        int n_send = offsets[rank + skips[k] + n_blocks] - offsets[rank + skips[k]];
        if (n_send) {
            int send_offset = offsets[rank + skips[k]] - offsets[rank];
            mpi_errno = MPIC_Isend(((char*) partial_buf__R) + (send_offset * extent), n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[0], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        }
        int n_recv = offsets[rank + n_blocks] - offsets[rank];
        if (n_recv) {
            mpi_errno = MPIC_Irecv(inter_buf__T, n_recv, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[(!!n_send)]);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (n_send || n_recv) {
            mpi_errno = MPIC_Waitall((!!n_send) + (!!n_recv), requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
        if (n_recv) {
            mpi_errno = MPIR_Reduce_local(inter_buf__T, partial_buf__R, n_recv, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    // Gather
    for (int k = 0; k < cga_tree_depth; k++) {
        int to = (rank - skips[k] + comm_size) % comm_size;
        int from = (rank + skips[k]) % comm_size;
        
        int n_blocks = skips[k+1] - skips[k];

        int n_send = offsets[rank + n_blocks] - offsets[rank];
        if (n_send) {
            mpi_errno = MPIC_Isend(partial_buf__R, n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[0], errflag);
            MPIR_ERR_CHECK(mpi_errno);
        }
        int n_recv = offsets[rank + skips[k] + n_blocks] - offsets[rank + skips[k]];
        if (n_recv) {
            int recv_offset = offsets[rank + skips[k]] - offsets[rank];
            mpi_errno = MPIC_Irecv(((char*) partial_buf__R) + (recv_offset * extent), n_recv, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[(!!n_send)]);
        }
        if (n_send || n_recv) {
            mpi_errno = MPIC_Waitall((!!n_send) + (!!n_recv), requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    // Undo rotation
    for (int i = 0; i < comm_size; i++) {
        int rel = rank + i;
        int b_count = offsets[rel+1] - offsets[rel];
        if (b_count != 0) {
            int src_offset = offsets[rel] - offsets[rank];
            int dst_offset = offsets[rel % comm_size];
            MPIR_Localcopy(((char*) partial_buf__R) + (src_offset * extent), b_count, datatype,
                           ((char*) recvbuf) + (dst_offset * extent), b_count, datatype);
        }
    }
    
    MPIR_CHKLMEM_FREEALL();

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
