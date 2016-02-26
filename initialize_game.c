
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "initialize_game.h"

static inline int rank_by_shift(
  const MPI_Comm communicator,
  const int* restrict const self_coords, const int* restrict const dims,
  const int dim0, const int dim1)
{

  const int coords[2] = { self_coords[0] + dim0, self_coords[1] + dim1 };
  int rank = MPI_PROC_NULL;

  if (coords[0] >= 0 && coords[0] < dims[0] &&
      coords[1] >= 0 && coords[1] < dims[1]) {
        MPI_Cart_rank(communicator, coords, &rank);
  }

  return rank;
}

// Inspired from:
// http://stackoverflow.com/questions/7549316/mpi-partition-matrix-into-blocks
static inline int scatter_matrix(
  const bool* restrict const global_matrix, const int global_rows, const int global_cols,
        bool* restrict const local_matrix , const int local_rows , const int local_cols ,
  const int* restrict const dims, const int sender_rank, const MPI_Comm communicator)
{
  int this_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

  // Define data type that will have appropiate block and stride length for
  // each sub matrix.
  MPI_Datatype send_blocktype;
  MPI_Datatype send_blocktype_vec;
  MPI_Type_vector(local_rows, local_cols, global_cols, MPI_C_BOOL, &send_blocktype_vec);
  MPI_Type_create_resized(send_blocktype_vec, 0, sizeof(bool), &send_blocktype);
  MPI_Type_commit(&send_blocktype);

  MPI_Datatype recv_blocktype;
  MPI_Datatype recv_blocktype_vec;
  MPI_Type_vector(local_rows, local_cols, local_cols + 2, MPI_C_BOOL, &recv_blocktype_vec);
  MPI_Type_create_resized(recv_blocktype_vec, 0, sizeof(bool), &recv_blocktype);
  MPI_Type_commit(&recv_blocktype);

  // Create displacement and count matrix for scatterv
  const int size = dims[0] * dims[1];
  int* restrict const disp = malloc(size * sizeof(int));
  int* restrict const count = malloc(size * sizeof(int));

  for (int rank = 0; rank < size; rank++) {
    int coords[2] = {0, 0};
    MPI_Cart_coords(communicator, rank, 2, coords);

    count[rank] = 1;
    disp[rank] = coords[0] * global_cols * local_rows + coords[1] * local_cols;

    if (this_rank == 0) {
      printf("(%d, %d) disp: %d\n", coords[0], coords[1], disp[rank]);
    }
  }

  // Scatter data
  int ierror = MPI_Scatterv(global_matrix, count, disp, send_blocktype,
                            &local_matrix[(local_cols + 2) + 1], 1, recv_blocktype,
                            sender_rank, communicator);

  // Free buffers and types
  free(disp);
  free(count);
  MPI_Type_free(&send_blocktype_vec);
  MPI_Type_free(&send_blocktype);
  MPI_Type_free(&recv_blocktype_vec);
  MPI_Type_free(&recv_blocktype);

  // Report errors
  return ierror;
}

int initialize_game(
  GameInfo* const game,
  const int size, const int rank,
  int rows, int cols, const bool* const restrict init)
{
  // Share cols and rows, these where loaded in rank 0
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Create cartesian topology
  int dims[2] = {0, 0}; // zero means not fixed, please replace
  int periods[2] = {0, 0}; // zero means not periodic
  int coords[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &game->communicator);
  MPI_Cart_coords(game->communicator, rank, 2, coords);

  // Ensure that the data can be spread evenly
  if (rows % dims[0] != 0 || cols % dims[1] != 0) {
    return 1;
  }

  // Set size properties
  game->global_rows = rows;
  game->global_cols = cols;
  game->local_rows = rows / dims[0];
  game->local_cols = cols / dims[1];

  // Set rank properties
  game->rank.north = rank_by_shift(game->communicator, coords, dims, -1, 0);
  game->rank.north_west = rank_by_shift(game->communicator, coords, dims, -1, -1);
  game->rank.north_east = rank_by_shift(game->communicator, coords, dims, -1, 1);

  game->rank.south = rank_by_shift(game->communicator, coords, dims, 1, 0);
  game->rank.south_west = rank_by_shift(game->communicator, coords, dims, 1, -1);
  game->rank.south_east = rank_by_shift(game->communicator, coords, dims, 1, 1);

  game->rank.east = rank_by_shift(game->communicator, coords, dims, 0, 1);
  game->rank.west = rank_by_shift(game->communicator, coords, dims, 0, -1);

  // allocate receive buffers
  game->recv.north = calloc(game->local_cols, sizeof(bool));
  game->recv.north_west = calloc(1, sizeof(bool));
  game->recv.north_east = calloc(1, sizeof(bool));

  game->recv.south = calloc(game->local_cols, sizeof(bool));
  game->recv.south_west = calloc(1, sizeof(bool));
  game->recv.south_east = calloc(1, sizeof(bool));

  game->recv.east = calloc(game->local_rows, sizeof(bool));
  game->recv.west = calloc(game->local_rows, sizeof(bool));

  // Calculate size of data array with 1 border padding
  int local_cols_pad = game->local_cols + 2;
  int local_rows_pad = game->local_rows + 2;

  // Allocate request and status buffers
  game->request = (MPI_Request*) malloc(16 * sizeof(MPI_Request));
  game->status = (MPI_Status*) malloc(16 * sizeof(MPI_Status));

  // Allocate game data buffers
  game->current = (bool*)calloc(local_rows_pad * local_cols_pad, sizeof(bool));
  game->previouse = (bool*)calloc(local_rows_pad * local_cols_pad, sizeof(bool));

  // Scatter initial data. This is just for distributing the loaded data,
  // not for handling boundery conditions.
  scatter_matrix(init, game->global_rows, game->global_cols,
                 game->current, game->local_rows , game->local_cols ,
                 dims, 0, game->communicator);

  return 0;
}

void destroy_game(GameInfo* const game) {
  // free communicator
  MPI_Comm_free(&game->communicator);

  // free receive buffers
  free(game->recv.north);
  free(game->recv.north_west);
  free(game->recv.north_east);

  free(game->recv.south);
  free(game->recv.south_west);
  free(game->recv.south_east);

  free(game->recv.east);
  free(game->recv.west);

  // free request and status buffers
  free(game->request);
  free(game->status);

  // free game data buffers
  free(game->current);
  free(game->previouse);
}
