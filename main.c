#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>

#include "debug_game.h"
#include "update_game.h"
#include "initialize_game.h"
#include "synchronize_game.h"

int main(int argc, char* argv[])
{
  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //
  // Initialization
  //
  GameInfo game;

  int rows = 0, cols = 0;
  bool* init = NULL;

  // rank 0 Loads data
  if (rank == 0)  {
    rows = 8;
    cols = 8;
    init = (bool[64]) {
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0
    };
  }

  // refactored
  if (initialize_game(&game, size, rank, rows, cols, init)) {
    return 1;
  }

  //printf("[%d] local matrix (%d x %d):\n", rank, game.local_rows + 2, game.local_cols + 2);
  //print_matrix(game.current, game.local_rows + 2, game.local_cols + 2);

  // perform iterations
  for (int iter = 0; iter < 5; iter++) {
    synchronize_game(&game);

    update_game(&game);

    // Gather the full game on rank 0 and print
    print_global_game(&game, rank);
  }

  // Free buffers
  destroy_game(&game);

  MPI_Finalize();
  return 0;
}
