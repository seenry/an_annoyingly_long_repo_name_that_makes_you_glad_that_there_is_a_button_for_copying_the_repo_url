#include "mpiimpl.h"
#include <math.h>

#define CHUNK_OFFSET(cid) (((cid) * chunk_size) + (((cid) < rrr_cutoff) ? (cid) : rrr_cutoff))

int MPIR_Allreduce_intra_circ_rrs_ag(const void* sendbuf,
                                       void* recvbuf,
                                       MPI_Aint count,
                                       MPI_Datatype datatype,
                                       MPI_Op op,
                                       MPIR_Comm* comm,
                                       int coll_attr)
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

    // Allocate memory
    MPIR_CHKLMEM_DECL();
    int* skips;
    MPIR_CHKLMEM_MALLOC(skips, (cga_tree_depth+1) * sizeof(int));
    void* intermed_buf;
    MPIR_CHKLMEM_MALLOC(intermed_buf, count * necessary_extent); // its possible to lower this footprint

    // Initialize coll data
    skips[cga_tree_depth] = comm_size;
    for (int i = cga_tree_depth-1; i >= 0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }

    int chunk_size = count/comm_size;
    int rrr_cutoff = count%comm_size;

    intermed_buf = (void*) ((char*) intermed_buf - true_lb);

    if (sendbuf != MPI_IN_PLACE) {
        MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
    }

    // Reduction
    MPIR_Request* requests[4];
    for (int k = cga_tree_depth - 1; k >= 0; k--) {
        int to = (rank + skips[k]) % comm_size;
        int from = (rank - skips[k] + comm_size) % comm_size;
        int n_req = 0;
        // Send logic
        int chunk0 = (rank + skips[k]) % comm_size;
        int chunk1 = (rank + skips[k+1] - 1) % comm_size;
        if (chunk0 <= chunk1) {
            // Contiguous Send
            int o0 = CHUNK_OFFSET(chunk0);
            int n_send = CHUNK_OFFSET(chunk1+1) - o0;
            if (n_send) {
                int send_offset = o0 * extent;
                mpi_errno = MPIC_Isend(((char*)recvbuf) + send_offset, n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else {
            // Wrapping Send
            int o0 = CHUNK_OFFSET(chunk0);
            int n_send0 = count - o0;
            if (n_send0) {
                int send_offset = o0 * extent;
                mpi_errno = MPIC_Isend(((char*)recvbuf) + send_offset, n_send0, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }

            int n_send1 = CHUNK_OFFSET(chunk1+1);
            if (n_send1) {
                mpi_errno = MPIC_Isend(recvbuf, n_send1, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }

        // Recv logic
        int rchunk0 = (from + skips[k]) % comm_size;
        int rchunk1 = (from + skips[k+1] - 1) % comm_size;
        int n_recv0 = 0;
        int n_recv1 = 0;
        if (rchunk0 <= rchunk1) {
            // Contiguous Recv
            n_recv0 = CHUNK_OFFSET(rchunk1+1) - CHUNK_OFFSET(rchunk0);
            if (n_recv0) {
                mpi_errno = MPIC_Irecv(intermed_buf, n_recv0, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else {
            // Wrapping Recv
            n_recv0 = count - CHUNK_OFFSET(rchunk0);
            if (n_recv0) {
                mpi_errno = MPIC_Irecv(intermed_buf, n_recv0, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }

            n_recv1 = CHUNK_OFFSET(rchunk1+1);
            if (n_recv1) {
                int recv_offset = n_recv0 * extent;
                mpi_errno = MPIC_Irecv(((char*)intermed_buf) + recv_offset, n_recv1, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }

        // Handle requests
        if (n_req) {
            mpi_errno = MPIC_Waitall(n_req, requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);

            if (n_recv0) {
                int recv_offset = CHUNK_OFFSET(rank) * extent;
                mpi_errno = MPIR_Reduce_local(intermed_buf, ((char*)recvbuf) + recv_offset, n_recv0, datatype, op);
                MPIR_ERR_CHECK(mpi_errno);
            }
            if (n_recv1) {
                int intermed_offset = n_recv0 * extent;
                mpi_errno = MPIR_Reduce_local(((char*)intermed_buf) + intermed_offset, recvbuf, n_recv1, datatype, op);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }
    }

    // Gather
    {
    int o0 = CHUNK_OFFSET(rank);
    for (int k = 0; k < cga_tree_depth; k++) {
        int to = (rank - skips[k] + comm_size) % comm_size;
        int from = (rank + skips[k]) % comm_size;
        int n_req = 0;
        
        // int chunk0 = rank;
        int chunk1 = (rank + skips[k+1] - skips[k] - 1) % comm_size;
        if (rank <= chunk1) {
            // Contiguous Send
            int n_send = CHUNK_OFFSET(chunk1+1) - o0;
            if (n_send) {
                int send_offset = o0 * extent;
                mpi_errno = MPIC_Isend(((char*)recvbuf) + send_offset, n_send, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else {
            // Wrapping Send
            int n_send0 = count - o0;
            if (n_send0) {
                int send_offset = o0 * extent;
                mpi_errno = MPIC_Isend(((char*)recvbuf) + send_offset, n_send0, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }

            int n_send1 = CHUNK_OFFSET(chunk1+1);
            if (n_send1) {
                mpi_errno = MPIC_Isend(recvbuf, n_send1, datatype, to, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++], coll_attr);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }

        int rchunk0 = from;
        int rchunk1 = (from + skips[k+1] - skips[k] - 1) % comm_size;
        if (rchunk0 <= rchunk1) {
            // Contiguous Recv
            int n_recv = CHUNK_OFFSET(rchunk1+1) - CHUNK_OFFSET(rchunk0);
            if (n_recv) {
                int recv_offset = CHUNK_OFFSET((rank + skips[k]) % comm_size) * extent;
                mpi_errno = MPIC_Irecv(((char*)recvbuf) + recv_offset, n_recv, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else {
            // Wrapping Recv
            int n_recv0 = count - CHUNK_OFFSET(rchunk0);
            if (n_recv0) {
                int recv_offset = CHUNK_OFFSET((rank + skips[k]) % comm_size) * extent;
                mpi_errno = MPIC_Irecv(((char*)recvbuf) + recv_offset, n_recv0, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }

            int n_recv1 = CHUNK_OFFSET(rchunk1 + 1);
            if (n_recv1) {
                mpi_errno = MPIC_Irecv(recvbuf, n_recv1, datatype, from, MPIR_ALLREDUCE_TAG, comm, &requests[n_req++]);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }

        if (n_req) {
            mpi_errno = MPIC_Waitall(n_req, requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }
    }

    MPIR_CHKLMEM_FREEALL();

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
