
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

// From http://stackoverflow.com/questions/10788180/sending-columns-of-a-matrix-using-mpi-scatter
// I'm getting that MPI_Type_create_resized is not needed for MPI_send MPI_recv
// calls.
/*
static inline void set_recv_row_type(
  struct TopologyDirection* const direction,
  const int offset, const int elements, const int cols)
{
  direction->recv_offset = offset;

  MPI_Type_vector(elements, 1, cols, MPI_C_BOOL, &direction->recv_type);
  MPI_Type_commit(&direction->recv_type);
}
*/
static inline void set_row_type(
  struct TopologyDirection* const send_direction, const int send_row,
  struct TopologyDirection* const recv_direction, const int recv_row,
  const int col_offset, const int elements, const int cols)
{
  send_direction->send_offset = col_offset + send_row * cols;
  recv_direction->recv_offset = col_offset + recv_row * cols;

  MPI_Type_contiguous(elements, MPI_C_BOOL, &send_direction->send_type);
  MPI_Type_commit(&send_direction->send_type);

  MPI_Type_contiguous(elements, MPI_C_BOOL, &recv_direction->recv_type);
  MPI_Type_commit(&recv_direction->recv_type);
}
static inline void set_row_type_new(
  struct TopologyDirectionNew* const send_direction, const int send_row,
  struct TopologyDirectionNew* const recv_direction, const int recv_row,
  const int col_offset, const int local_rows, const int local_cols)
{
  // Create subarray send data type
  int array_size[]        = {local_rows + 2, local_cols + 2};
  int array_subsize[]     = {1,              local_cols};
  int array_offset_send[] = {send_row,       col_offset};

  MPI_Type_create_subarray(2, array_size, array_subsize, array_offset_send,
                           MPI_ORDER_C, MPI_C_BOOL, &send_direction->send_type);
  MPI_Type_commit(&send_direction->send_type);

  // Create subarray recv data type
  int array_offset_recv[] = {recv_row,       col_offset};
  MPI_Type_create_subarray(2, array_size, array_subsize, array_offset_recv,
                           MPI_ORDER_C, MPI_C_BOOL, &recv_direction->recv_type);
  MPI_Type_commit(&recv_direction->recv_type);
}


static inline void set_corner_type(
  struct TopologyDirection* const send_direction, const int send_row, const int send_col_offset,
  struct TopologyDirection* const recv_direction, const int recv_row, const int recv_col_offset,
  const int cols)
{
  send_direction->send_offset = send_col_offset + send_row * cols;
  recv_direction->recv_offset = recv_col_offset + recv_row * cols;

  MPI_Type_contiguous(1, MPI_C_BOOL, &send_direction->send_type);
  MPI_Type_commit(&send_direction->send_type);

  MPI_Type_contiguous(1, MPI_C_BOOL, &recv_direction->recv_type);
  MPI_Type_commit(&recv_direction->recv_type);
}

static inline void set_col_type(
  struct TopologyDirection* const send_direction, const int send_col,
  struct TopologyDirection* const recv_direction, const int recv_col,
  const int row_offset, const int elements, const int cols)
{
  send_direction->send_offset = send_col + row_offset * cols;
  recv_direction->recv_offset = recv_col + row_offset * cols;

  MPI_Type_vector(elements, 1, cols, MPI_C_BOOL, &send_direction->send_type);
  MPI_Type_commit(&send_direction->send_type);

  MPI_Type_vector(elements, 1, cols, MPI_C_BOOL, &recv_direction->recv_type);
  MPI_Type_commit(&recv_direction->recv_type);
}
static inline void set_col_type_new(
  struct TopologyDirection* const send_direction, const int send_col,
  struct TopologyDirection* const recv_direction, const int recv_col,
  const int row_offset, const int local_rows, const int local_cols)
{
  // Create subarray send data type
  int array_size[]        = {local_rows + 2, local_cols + 2};
  int array_subsize[]     = {local_cols,     1};
  int array_offset_send[] = {row_offset,     send_col};

  MPI_Type_create_subarray(2, array_size, array_subsize, array_offset_send,
                           MPI_ORDER_C, MPI_C_BOOL, &send_direction->send_type);
  MPI_Type_commit(&send_direction->send_type);

  // Create subarray recv data type
  int array_offset_recv[] = {row_offset,     recv_col};
  MPI_Type_create_subarray(2, array_size, array_subsize, array_offset_recv,
                           MPI_ORDER_C, MPI_C_BOOL, &recv_direction->recv_type);
  MPI_Type_commit(&recv_direction->recv_type);
}

// Inspired from:
// http://stackoverflow.com/questions/7549316/mpi-partition-matrix-into-blocks
static inline int scatter_matrix(
  const bool* restrict const global_matrix, const int global_rows, const int global_cols,
        bool* restrict const local_matrix , const int local_rows , const int local_cols ,
  const int* restrict const dims, const int sender_rank, const MPI_Comm communicator)
{
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
  int node_dims[2] = {0, 0}; // zero means not fixed, please replace
  int periods[2] = {0, 0}; // zero means not periodic
  int coords[2] = {0, 0};
  MPI_Dims_create(size, 2, node_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, node_dims, periods, 1, &game->communicator);
  MPI_Cart_coords(game->communicator, rank, 2, coords);

  // Ensure that the data can be spread evenly
  if (rows % node_dims[0] != 0 || cols % node_dims[1] != 0) {
    return 1;
  }

  // Set size properties
  game->node_dims[0] = node_dims[0];
  game->node_dims[1] = node_dims[1];
  game->global_rows = rows;
  game->global_cols = cols;
  game->local_rows = rows / node_dims[0];
  game->local_cols = cols / node_dims[1];

  // Set rank properties
  game->topology.north.rank = rank_by_shift(game->communicator, coords, node_dims, -1, 0);
  game->topology.north_west.rank = rank_by_shift(game->communicator, coords, node_dims, -1, -1);
  game->topology.north_east.rank = rank_by_shift(game->communicator, coords, node_dims, -1, 1);

  game->topology.south.rank = rank_by_shift(game->communicator, coords, node_dims, 1, 0);
  game->topology.south_west.rank = rank_by_shift(game->communicator, coords, node_dims, 1, -1);
  game->topology.south_east.rank = rank_by_shift(game->communicator, coords, node_dims, 1, 1);

  game->topology.east.rank = rank_by_shift(game->communicator, coords, node_dims, 0, 1);
  game->topology.west.rank = rank_by_shift(game->communicator, coords, node_dims, 0, -1);

  // Create stride types
  // north -> south
  /*set_row_type(&game->topology.north, 1,
               &game->topology.south, game->local_rows + 1,
               1, game->local_cols, game->local_cols + 2);*/
  set_row_type_new(&game->topology.north, 1,
                   &game->topology.south, game->local_rows + 1,
                   1, game->local_rows, game->local_cols);


  // north west -> south east
  set_corner_type(&game->topology.north_west, 1                   , 1,
                  &game->topology.south_east, game->local_rows + 1, game->local_cols + 1,
                  game->local_cols + 2);
  // north east -> south west
  set_corner_type(&game->topology.north_east, 1                   , game->local_cols,
                  &game->topology.south_west, game->local_rows + 1, 0,
                  game->local_cols + 2);

  // south -> north
  /*set_row_type(&game->topology.south, game->local_rows,
               &game->topology.north, 0,
               1, game->local_cols, game->local_cols + 2);*/
  set_row_type_new(&game->topology.south, game->local_rows,
                   &game->topology.north, 0,
                   1, game->local_rows, game->local_cols);


  // south west -> north east
  set_corner_type(&game->topology.south_west, game->local_rows, 1,
                  &game->topology.north_east, 0               , game->local_cols + 1,
                  game->local_cols + 2);
  // south east -> north west
  set_corner_type(&game->topology.south_east, game->local_rows, game->local_cols,
                  &game->topology.north_west, 0               , 0,
                  game->local_cols + 2);

  // east -> west
  /*set_col_type(&game->topology.east, game->local_cols,
               &game->topology.west, 0,
               1, game->local_rows, game->local_cols + 2);*/

  set_col_type_new(&game->topology.east, game->local_cols,
              &game->topology.west, 0,
              1, game->local_rows, game->local_cols);
  // west -> east
  /*set_col_type(&game->topology.west, 1,
               &game->topology.east, game->local_cols + 1,
               1, game->local_rows, game->local_cols + 2);*/
  set_col_type(&game->topology.west, 1,
              &game->topology.east, game->local_cols + 1,
              1, game->local_rows, game->local_cols);

  // Allocate request and status buffers
  game->request = (MPI_Request*) malloc(16 * sizeof(MPI_Request));
  game->status = (MPI_Status*) malloc(16 * sizeof(MPI_Status));

  // Allocate game data buffers
  int full_size = (game->local_cols + 2) * (game->local_rows + 2);
  game->current = (bool*)calloc(full_size, sizeof(bool));
  game->previouse = (bool*)calloc(full_size, sizeof(bool));

  // Scatter initial data. This is just for distributing the loaded data,
  // not for handling boundery conditions.
  scatter_matrix(init, game->global_rows, game->global_cols,
                 game->current, game->local_rows , game->local_cols ,
                 node_dims, 0, game->communicator);

  return 0;
}

static inline void destroy_direction_struct(struct TopologyDirection* direction) {
  MPI_Type_free(&direction->send_type);
  MPI_Type_free(&direction->recv_type);
}
static inline void destroy_direction_struct_new(struct TopologyDirectionNew* direction) {
  MPI_Type_free(&direction->send_type);
  MPI_Type_free(&direction->recv_type);
}


void destroy_game(GameInfo* const game) {
  // free communicator
  MPI_Comm_free(&game->communicator);

  // free topology direction stucts
  //destroy_direction_struct(&game->topology.north);
  destroy_direction_struct_new(&game->topology.north);
  destroy_direction_struct(&game->topology.north_west);
  destroy_direction_struct(&game->topology.north_east);

  //destroy_direction_struct(&game->topology.south);
  destroy_direction_struct_new(&game->topology.south);
  destroy_direction_struct(&game->topology.south_west);
  destroy_direction_struct(&game->topology.south_east);

  destroy_direction_struct(&game->topology.east);
  destroy_direction_struct(&game->topology.west);

  // free request and status buffers
  free(game->request);
  free(game->status);

  // free game data buffers
  free(game->current);
  free(game->previouse);
}
