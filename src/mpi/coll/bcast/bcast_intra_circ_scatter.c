#include "mpiimpl.h"

int MPIR_Bcast_intra_circ_scatter(void *buffer,
                                  MPI_Aint count,
                                  MPI_Datatype datatype,
                                  int root, MPIR_Comm* comm,
                                  int coll_attr)
{
    int mpi_errno = MPI_SUCCESS;

    int comm_size = comm->local_size;
    int rank = comm->rank;

    if (comm_size < 2) {
        goto fn_exit;
    }
    
    int depth = 1;
    while (0x1<<depth < comm_size) {
        depth++;
    }

    MPIR_CHKLMEM_DECL();
    int* skips; char* do_send; char* do_recv;
    MPIR_CHKLMEM_MALLOC(skips, (depth + 1) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(do_recv, depth * sizeof(char));
    MPIR_CHKLMEM_MALLOC(do_send, depth * sizeof(char));
    
    // Precalculate skips
    skips[depth] = comm_size;
    for (int i = depth - 1; i >= 0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }

    // Figure out send/recvs
	for (int i = 0; i < depth; i++) {
        do_recv[i] = 0;
        do_send[i] = 0;
    }

    int rel_rank = (rank - root + comm_size) % comm_size;
    for (int dst_rank = 1; dst_rank < comm_size; dst_rank++) {
        int intermed = dst_rank;
        for (int k = depth-1; k >= 0; k--) {
            if (intermed - skips[k] >= 0) {
                if (intermed == rel_rank) {
                    do_recv[k] = 1;
                    break;
                }
                intermed -= skips[k];
                if (intermed == rel_rank) {
                    do_send[k] = 1;
                }
            }
        }
    }

    // BCast
    MPIR_Request* requests[1];
    {
    int i = 0;
    if (rank != root) {
        for (; i < depth; i++) {
            if (do_recv[i]) {
                int from = (rank - skips[i] + comm_size) % comm_size;
                mpi_errno = MPIC_Irecv(buffer, count, datatype, from, MPIR_BCAST_TAG, comm, &requests[0]);
                MPIR_ERR_CHECK(mpi_errno);
                mpi_errno = MPIC_Waitall(1, requests, MPI_STATUSES_IGNORE);
                MPIR_ERR_CHECK(mpi_errno);
                i++;
                break;
            }
        }
    }
    for (; i < depth; i++) {
        if (do_send[i]) {
            int to = (rank + skips[i]) % comm_size;
            mpi_errno = MPIC_Isend(buffer, count, datatype, to, MPIR_BCAST_TAG, comm, &requests[0], coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIC_Waitall(1, requests, MPI_STATUSES_IGNORE);
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

