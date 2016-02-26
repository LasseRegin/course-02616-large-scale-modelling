#include <stdio.h>
#include <stdlib.h>

#include "initialize_game.h"

void print_matrix(const bool* const restrict game, const int rows, const int cols) {
  char digits[] = {'0', '1'};

  char* outdata = (char*)malloc(rows * (cols * 2 + 1) + 2 * sizeof(char));
  int outindex = 0;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      outdata[outindex++] = digits[ game[r * cols + c] ];
      outdata[outindex++] = ' ';
    }
    outdata[outindex++] = '\n';
  }
  outdata[outindex++] = '\n';
  outdata[outindex++] = '\0';

  printf("%s", outdata);
  free(outdata);
}

static inline int gatter_matrix(
  const bool* restrict const local_matrix , const int local_rows , const int local_cols ,
        bool* restrict const global_matrix, const int global_rows, const int global_cols,
  const int* restrict const dims, const int recv_rank, const MPI_Comm communicator)
{
  // Define data type that will have appropiate block and stride length for
  // each sub matrix.
  MPI_Datatype send_blocktype;
  MPI_Datatype send_blocktype_vec;
  MPI_Type_vector(local_rows, local_cols, local_cols + 2, MPI_C_BOOL, &send_blocktype_vec);
  MPI_Type_create_resized(send_blocktype_vec, 0, sizeof(bool), &send_blocktype);
  MPI_Type_commit(&send_blocktype);

  MPI_Datatype recv_blocktype;
  MPI_Datatype recv_blocktype_vec;
  MPI_Type_vector(local_rows, local_cols, global_cols, MPI_C_BOOL, &recv_blocktype_vec);
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
  int ierror = MPI_Gatherv(&local_matrix[(local_cols + 2) + 1], 1, send_blocktype,
                           global_matrix, count, disp, recv_blocktype,
                           recv_rank, communicator);

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

void print_global_game(GameInfo* game, const int rank) {
  // Gather the full game on rank 0
  bool* full_game = NULL;
  if (rank == 0) {
    full_game = (bool*)calloc(game->global_rows * game->global_cols, sizeof(bool));
  }

  gatter_matrix(game->current, game->local_rows, game->local_cols,
                full_game, game->global_rows, game->global_cols,
                game->node_dims, 0, game->communicator);

  // rank 0 print full game and free
  if (rank == 0) {
    printf("full_game:\n");
    print_matrix(full_game, game->global_rows, game->global_cols);
    fflush(stdout);
    free(full_game);
  }
}
