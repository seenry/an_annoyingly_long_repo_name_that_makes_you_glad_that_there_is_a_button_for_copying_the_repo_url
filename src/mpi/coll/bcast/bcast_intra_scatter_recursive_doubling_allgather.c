/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "bcast.h"

/* Algorithm: Broadcast based on a scatter followed by an allgather.
 *
 * We first scatter the buffer using a binomial tree algorithm. This costs
 * lgp.alpha + n.((p-1)/p).beta
 * If the datatype is contiguous, we treat the data as bytes and
 * divide (scatter) it among processes by using ceiling division.
 * For the noncontiguous, we first pack the data into a temporary
 * buffer by using MPI_Pack, scatter it as bytes, and unpack it
 * after the allgather.
 *
 * For the allgather, we use a recursive doubling algorithm for
 * medium-size messages and power-of-two number of processes. This
 * takes lgp steps. In each step pairs of processes exchange all the
 * data they have (we take care of non-power-of-two situations). This
 * costs approximately lgp.alpha + n.((p-1)/p).beta. (Approximately
 * because it may be slightly more in the non-power-of-two case, but
 * it's still a logarithmic algorithm.) Therefore, for long messages
 * Total Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta
*/
int MPIR_Bcast_intra_scatter_recursive_doubling_allgather(void *buffer,
                                                          MPI_Aint count,
                                                          MPI_Datatype datatype,
                                                          int root,
                                                          MPIR_Comm * comm_ptr, int coll_attr)
{
    MPI_Status status;
    int rank, comm_size, dst;
    int relative_rank, mask;
    int mpi_errno = MPI_SUCCESS;
    MPI_Aint curr_size, recv_size = 0;
    int j, k, i, tmp_mask, is_contig;
    MPI_Aint type_size, nbytes;
    int relative_dst, dst_tree_root, my_tree_root;
    int tree_root, nprocs_completed;
    MPIR_CHKLMEM_DECL();
    MPI_Aint true_extent, true_lb;
    void *tmp_buf;

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);
    relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

    if (HANDLE_IS_BUILTIN(datatype))
        is_contig = 1;
    else {
        MPIR_Datatype_is_contig(datatype, &is_contig);
    }

    MPIR_Datatype_get_size_macro(datatype, type_size);

    nbytes = type_size * count;
    if (nbytes == 0)
        goto fn_exit;   /* nothing to do */

    if (is_contig) {
        /* contiguous. no need to pack. */
        MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);

        tmp_buf = MPIR_get_contig_ptr(buffer, true_lb);
    } else {
        MPIR_CHKLMEM_MALLOC(tmp_buf, nbytes);

        if (rank == root) {
            mpi_errno =
                MPIR_Localcopy(buffer, count, datatype, tmp_buf, nbytes, MPIR_BYTE_INTERNAL);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    MPI_Aint scatter_size;
    scatter_size = (nbytes + comm_size - 1) / comm_size;        /* ceiling division */

    mpi_errno = MPII_Scatter_for_bcast(buffer, count, datatype, root, comm_ptr,
                                       nbytes, tmp_buf, is_contig, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    /* curr_size is the amount of data that this process now has stored in
     * buffer at byte offset (relative_rank*scatter_size) */
    /* Note: since we are rounding up scatter_size, higher ranks may not have data and nbytes-offset may be negative */
    curr_size = MPL_MIN(scatter_size, (nbytes - (relative_rank * scatter_size)));
    if (curr_size < 0)
        curr_size = 0;

    /* medium size allgather and pof2 comm_size. use recursive doubling. */

    mask = 0x1;
    i = 0;
    while (mask < comm_size) {
        relative_dst = relative_rank ^ mask;

        dst = (relative_dst + root) % comm_size;

        /* find offset into send and recv buffers.
         * zero out the least significant "i" bits of relative_rank and
         * relative_dst to find root of src and dst
         * subtrees. Use ranks of roots as index to send from
         * and recv into  buffer */

        dst_tree_root = relative_dst >> i;
        dst_tree_root <<= i;

        my_tree_root = relative_rank >> i;
        my_tree_root <<= i;

        MPI_Aint send_offset, recv_offset;
        send_offset = my_tree_root * scatter_size;
        recv_offset = dst_tree_root * scatter_size;

        if (relative_dst < comm_size) {
            mpi_errno = MPIC_Sendrecv(((char *) tmp_buf + send_offset),
                                      curr_size, MPIR_BYTE_INTERNAL, dst, MPIR_BCAST_TAG,
                                      ((char *) tmp_buf + recv_offset),
                                      (nbytes - recv_offset < 0 ? 0 : nbytes - recv_offset),
                                      MPIR_BYTE_INTERNAL, dst, MPIR_BCAST_TAG, comm_ptr, &status,
                                      coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            if (mpi_errno) {
                recv_size = 0;
            } else
                MPIR_Get_count_impl(&status, MPIR_BYTE_INTERNAL, &recv_size);
            curr_size += recv_size;
        }

        /* if some processes in this process's subtree in this step
         * did not have any destination process to communicate with
         * because of non-power-of-two, we need to send them the
         * data that they would normally have received from those
         * processes. That is, the haves in this subtree must send to
         * the havenots. We use a logarithmic recursive-halfing algorithm
         * for this. */

        /* This part of the code will not currently be
         * executed because we are not using recursive
         * doubling for non power of two. Mark it as experimental
         * so that it doesn't show up as red in the coverage tests. */

        /* --BEGIN EXPERIMENTAL-- */
        if (dst_tree_root + mask > comm_size) {
            nprocs_completed = comm_size - my_tree_root - mask;
            /* nprocs_completed is the number of processes in this
             * subtree that have all the data. Send data to others
             * in a tree fashion. First find root of current tree
             * that is being divided into two. k is the number of
             * least-significant bits in this process's rank that
             * must be zeroed out to find the rank of the root */
            j = mask;
            k = 0;
            while (j) {
                j >>= 1;
                k++;
            }
            k--;

            MPI_Aint offset;
            offset = scatter_size * (my_tree_root + mask);
            tmp_mask = mask >> 1;

            while (tmp_mask) {
                relative_dst = relative_rank ^ tmp_mask;
                dst = (relative_dst + root) % comm_size;

                tree_root = relative_rank >> k;
                tree_root <<= k;

                /* send only if this proc has data and destination
                 * doesn't have data. */

                /* if (rank == 3) {
                 * printf("rank %d, dst %d, root %d, nprocs_completed %d\n", relative_rank, relative_dst, tree_root, nprocs_completed);
                 * fflush(stdout);
                 * } */

                if ((relative_dst > relative_rank) && (relative_rank < tree_root + nprocs_completed)
                    && (relative_dst >= tree_root + nprocs_completed)) {

                    /* printf("Rank %d, send to %d, offset %d, size %d\n", rank, dst, offset, recv_size);
                     * fflush(stdout); */
                    mpi_errno = MPIC_Send(((char *) tmp_buf + offset),
                                          recv_size, MPIR_BYTE_INTERNAL, dst,
                                          MPIR_BCAST_TAG, comm_ptr, coll_attr);
                    /* recv_size was set in the previous
                     * receive. that's the amount of data to be
                     * sent now. */
                    MPIR_ERR_CHECK(mpi_errno);
                }
                /* recv only if this proc. doesn't have data and sender
                 * has data */
                else if ((relative_dst < relative_rank) &&
                         (relative_dst < tree_root + nprocs_completed) &&
                         (relative_rank >= tree_root + nprocs_completed)) {
                    /* printf("Rank %d waiting to recv from rank %d\n",
                     * relative_rank, dst); */
                    mpi_errno = MPIC_Recv(((char *) tmp_buf + offset),
                                          nbytes - offset < 0 ? 0 : nbytes - offset,
                                          MPIR_BYTE_INTERNAL, dst, MPIR_BCAST_TAG, comm_ptr,
                                          &status);
                    /* nprocs_completed is also equal to the no. of processes
                     * whose data we don't have */
                    MPIR_ERR_CHECK(mpi_errno);
                    if (mpi_errno) {
                        recv_size = 0;
                    } else
                        MPIR_Get_count_impl(&status, MPIR_BYTE_INTERNAL, &recv_size);
                    curr_size += recv_size;
                    /* printf("Rank %d, recv from %d, offset %d, size %d\n", rank, dst, offset, recv_size);
                     * fflush(stdout); */
                }
                tmp_mask >>= 1;
                k--;
            }
        }
        /* --END EXPERIMENTAL-- */

        mask <<= 1;
        i++;
    }

#ifdef HAVE_ERROR_CHECKING
    /* check that we received as much as we expected */
    MPIR_ERR_CHKANDJUMP2(curr_size != nbytes, mpi_errno, MPI_ERR_OTHER,
                         "**collective_size_mismatch",
                         "**collective_size_mismatch %d %d", (int) curr_size, (int) nbytes);

#endif

    if (!is_contig) {
        if (rank != root) {
            mpi_errno =
                MPIR_Localcopy(tmp_buf, nbytes, MPIR_BYTE_INTERNAL, buffer, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
