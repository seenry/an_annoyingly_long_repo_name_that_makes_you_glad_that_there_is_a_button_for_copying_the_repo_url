#include "mpiimpl.h"

struct sched_args_t {
    int* skips;
    int* send_sched;
    int* next;
    int* prev;
    int* extra;
    int tree_depth;
    int comm_size;
};

int all_blocks(int r, int r_, int s, int e, int k, int* buffer, struct sched_args_t* args);
void gen_rsched(int r, int* buffer, struct sched_args_t* args);
void gen_ssched(int r, struct sched_args_t* args);

int get_baseblock(int r, struct sched_args_t* args);

int MPIR_Bcast_intra_circ_vring(void *buffer,
                                MPI_Aint count,
                                MPI_Datatype datatype,
                                int root, MPIR_Comm* comm,
                                const int chunk_size, int coll_attr)
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
    int* skips; int* recv_sched; int* send_sched; int* next; int* prev; int* extra;
    MPIR_CHKLMEM_MALLOC(skips, (depth + 1) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(recv_sched, depth * sizeof(int));
    MPIR_CHKLMEM_MALLOC(send_sched, depth * sizeof(int));
    MPIR_CHKLMEM_MALLOC(next, (depth + 2) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(prev, (depth + 2) * sizeof(int));
    MPIR_CHKLMEM_MALLOC(extra, depth * sizeof(int));
    
    // Precalculate skips
    //   (hard to do at runtime during the `all_blocks` subroutine)
    skips[depth] = comm_size;
    for (int i = depth - 1; i >= 0; i--) {
        skips[i] = (skips[i+1] / 2) + (skips[i+1] & 0x1);
    }

    // Generate schedules
    struct sched_args_t args = {
        skips, send_sched, next + 1, prev + 1, extra,
        depth, comm_size
    };
    gen_rsched(rank, recv_sched, &args);
    gen_ssched(rank, &args);

    // Datatype Handling:
    MPI_Aint type_size;
    int is_contig;
    int buf_size;

    MPIR_Datatype_get_size_macro(datatype, type_size);
    buf_size = count * type_size;

    // MPI_Aint true_extent, true_lb, extent;
    // MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    // MPIR_Datatype_get_extent_macro(datatype, extent);
    // printf("%d %d %d\n", type_size, extent, true_extent);

    if (HANDLE_IS_BUILTIN(datatype))
        is_contig = 1;
    else {
        MPIR_Datatype_is_contig(datatype, &is_contig);
    }
    void* tmp_buf;
    if (!is_contig) {
        MPIR_CHKLMEM_MALLOC(tmp_buf, buf_size);
        if (rank == root) {
            mpi_errno = MPIR_Localcopy(buffer, count, datatype, tmp_buf, buf_size, MPIR_BYTE_INTERNAL);
            MPIR_ERR_CHECK(mpi_errno);
        }
    } else {
        tmp_buf = buffer;
    }
    
    // Handle Chunking
    int n_chunk;
    int last_msg_size;

    if (chunk_size == 0) {
        n_chunk = 1;
        last_msg_size = buf_size;
    } else {
        n_chunk = (buf_size + chunk_size - 1) / chunk_size;
        last_msg_size = (buf_size % chunk_size == 0)
                      ? chunk_size
                      : buf_size % chunk_size;
    }

    // Run schedule
    int x = (((depth - ((n_chunk - 1) % depth)) % depth) + depth) % depth;
    int offset = -x;
    MPIR_Request* requests[1];
    for (int i = x; i < n_chunk - 1 + depth + x; i++) {
        int k = i % depth;

        if (send_sched[k] + offset >= 0) {
            int peer = (rank + skips[k]) % comm_size;
            
            int send_block = send_sched[k] + offset;
            if (send_block >= n_chunk) send_block = n_chunk - 1;
            int msg_size = (send_block != n_chunk - 1) ? chunk_size : last_msg_size;

            mpi_errno = MPIC_Isend((char*) tmp_buf + (chunk_size * send_block), msg_size, MPIR_BYTE_INTERNAL, peer, MPIR_BCAST_TAG, comm, &requests[0], coll_attr);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIC_Waitall(1, requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        } else if (recv_sched[k] + offset >= 0) {
            int peer = (rank - skips[k] + comm_size) % comm_size;
            
            int recv_block = recv_sched[k] + offset;
            if (recv_block >= n_chunk) recv_block = n_chunk - 1;
            int msg_size = (recv_block != n_chunk - 1) ? chunk_size : last_msg_size;

            mpi_errno = MPIC_Irecv((char*) tmp_buf + (chunk_size * recv_block), msg_size, MPIR_BYTE_INTERNAL, peer, MPIR_BCAST_TAG, comm, &requests[0]);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIC_Waitall(1, requests, MPI_STATUSES_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }

        if (k == depth - 1) {
            offset += depth;
        }
    }

    if (!is_contig) {
        if (rank != root) {
            mpi_errno = MPIR_Localcopy(tmp_buf, buf_size, MPIR_BYTE_INTERNAL, buffer, count, datatype);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }
    
    MPIR_CHKLMEM_FREEALL();

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

//////// HELPER FUNCTIONS ////////

int all_blocks(int r, int r_, int s, int e, int k, int* buffer, struct sched_args_t* args) {
    while (e != -1) {
        if ((r_ + args->skips[e] <= r - args->skips[k])
         && (r_ + args->skips[e] < s)) {
            if (r_ + args->skips[e] <= r - args->skips[k+1]) {
                k = all_blocks(r, r_ + args->skips[e], s, e, k, buffer, args);
            }
            if (r_ > r - args->skips[k+1]) {
                return k;
            }
            s = r_ + args->skips[e];
            buffer[k] = e;
            k += 1;
            args->next[args->prev[e]] = args->next[e];
            args->prev[args->next[e]] = args->prev[e];
        }
        e = args->next[e];
    }
    return k;
}

void gen_rsched(int r, int* buffer, struct sched_args_t* args) {
    for (int i = 0; i <= args->tree_depth; i++) {
        args->next[i] = i - 1;
        args->prev[i] = i + 1;
    }
    args->prev[args->tree_depth] = -1;
    args->next[-1] = args->tree_depth;
    args->prev[-1] = 0;

    int b = get_baseblock(r, args);
    
    args->next[args->prev[b]] = args->next[b];
    args->prev[args->next[b]] = args->prev[b];
    
    all_blocks(args->comm_size + r, 0, args->comm_size * 2, args->tree_depth, 0, buffer, args);
    
    for (int i = 0; i < args->tree_depth; i++) {
        if (buffer[i] == args->tree_depth) {
            buffer[i] = b;
        } else {
            buffer[i] = buffer[i] - args->tree_depth;
        }
    }
}

void gen_ssched(int r, struct sched_args_t* args) {
    if (r == 0) {
        for (int i = 0; i < args->tree_depth; i++) {
            args->send_sched[i] = i;
        }
        return;
    }

    int b = get_baseblock(r, args);

    int r_ = r;
    int c = b;
    int e = args->comm_size;
    for (int i = args->tree_depth - 1; i > 0; i--) {
        if (r_ < args->skips[i]) {
            if ((r_ + args->skips[i] < e)
             || (e < args->skips[i-1])
             || ((i == 1)
              && (b > 0))) {
                args->send_sched[i] = c;
            } else {
                gen_rsched((r + args->skips[i]) % args->comm_size, args->extra, args);
                args->send_sched[i] = args->extra[i];
            }
            if (e > args->skips[i]) {
                e = args->skips[i];
            }
        } else {
            c = i - args->tree_depth;
            e = e - args->skips[i];
            if ((r_ > args->skips[i])
             || (r_ <= e)
             || (i == 1)
             || (e < args->skips[i-1])) {
                args->send_sched[i] = c;
            } else {
                gen_rsched((r + args->skips[i]) % args->comm_size, args->extra, args);
                args->send_sched[i] = args->extra[i];
            }
            r_ -= args->skips[i];
        }
    }
    args->send_sched[0] = b - args->tree_depth;
}

int get_baseblock(int r, struct sched_args_t* args) {
    int r_ = 0;
    for (int i = args->tree_depth - 1; i >= 0; i--) {
        if (r_ + args->skips[i] == r) {
            return i;
        } else if (r_ + args->skips[i] < r) {
            r_ += args->skips[i];
        }
    }
    return args->tree_depth;
}
