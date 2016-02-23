#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "update_game.h"

static const int SEND_NORTH_TAG = 10;
static const int SEND_SOUTH_TAG = 11;

static inline void swap_arrays(bool** game_a, bool** game_b)
{
  bool* temp = *game_a;
  *game_a = *game_b;
  *game_b = temp;
}

static inline void print_matrix(const bool* const restrict game, const int rows, const int cols) {
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

int main(int argc, char* argv[])
{
	MPI_Init(&argc,&argv);

	int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //
  // Initialization
  //

  int N;
  bool* init = NULL;

  // rank 0 Loads data
  if (rank == 0)  {
    N = 8;
    init = (bool[64]) {
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
    };
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Ensure that the data can be spread evenly
  if (N % size != 0) {
    return 1;
  }

  // Calculate the number of rows each process is responsibol for
  int N_pad = N + 2;
  int rows = N / size;

  // Allocate buffers
  bool* game_a = (bool*)calloc((rows + 2) * N_pad, sizeof(bool));
  bool* game_b = (bool*)calloc((rows + 2) * N_pad, sizeof(bool));

  // Scatter initial data. This is just for distributing the loaded data,
  // not for handling boundery conditions.
  bool* init_partial = (bool*)calloc(rows * N, sizeof(bool));
  MPI_Scatter(init, rows * N, MPI_C_BOOL, init_partial, rows * N, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  // Move `init_partial` into the game buffer where the boundaries are considered
  for (int init_row = 0; init_row < rows; init_row++) {
    int game_row = init_row + 1;

    for (int init_col = 0; init_col < N; init_col++) {
      int game_col = init_col + 1;

      game_a[game_row * N_pad + game_col] = init_partial[init_row * N + init_col];
    }
  }
  free(init_partial);

  //
  // Iteration prepreations
  //

  // Define north and south ranks
  int north_rank = rank - 1;
  int south_rank = rank + 1;
  if (rank == 0) north_rank = MPI_PROC_NULL;
  if (rank == size - 1) south_rank = MPI_PROC_NULL;

  // Allocate boundary communication buffers
  bool* north_recv = (bool*)calloc(N, sizeof(bool));
  bool* south_recv = (bool*)calloc(N, sizeof(bool));
  MPI_Request* request = (MPI_Request*) malloc(4 * sizeof(MPI_Request));
  MPI_Status* status = (MPI_Status*) malloc(4 * sizeof(MPI_Status));

  for (int iter = 0; iter < 5; iter++) {
    //
    // Transfer boundary conditions
    //

    // send data to north, receive from south
    MPI_Isend(&game_a[N_pad + 1], N, MPI_C_BOOL, north_rank, SEND_NORTH_TAG, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(south_recv, N, MPI_C_BOOL, south_rank, SEND_NORTH_TAG, MPI_COMM_WORLD, &request[1]);

    // send data to south, receive from north
    MPI_Isend(&game_a[rows * N_pad + 1], N, MPI_C_BOOL, south_rank, SEND_SOUTH_TAG, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(north_recv, N, MPI_C_BOOL, north_rank, SEND_SOUTH_TAG, MPI_COMM_WORLD, &request[3]);

    // await completion
    MPI_Waitall(4, request, status);

    // transfer receive buffers to game buffers
    memcpy(&game_a[1], north_recv, N * sizeof(bool));
    memcpy(&game_a[(rows + 1) * N_pad + 1], south_recv, N * sizeof(bool));

    //
    // Update game
    //
    update_game(game_a, game_b, rows, N);
    swap_arrays(&game_a, &game_b);

    //
    // Debug
    //

    // Gather the full game on rank 0
    bool* full_game = NULL;
    if (rank == 0) {
      full_game = (bool*)calloc(N_pad * N_pad, sizeof(bool));
    }
    MPI_Gather(&game_a[N_pad], N_pad * rows, MPI_C_BOOL, &full_game[N_pad], N_pad * rows, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // rank 0 print full game and free
    if (rank == 0) {
      printf("full_game:\n");
      print_matrix(full_game, N_pad, N_pad);
      free(full_game);
    }
  }

  // Free communication buffers
  free(north_recv);
  free(south_recv);
  free(request);
  free(status);

  // Free game buffers
  free(game_a);
  free(game_b);

	MPI_Finalize();
  return 0;
}
