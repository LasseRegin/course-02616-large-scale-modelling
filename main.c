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
  GameCommunicator communicator;

  int N = 0;
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

  if (initialize_game(&game, size, N, init)) {
    return 1;
  }

  initialize_communicator(&communicator, &game, size, rank);

	// perform iterations
  for (int iter = 0; iter < 5; iter++) {
    synchronize_game(&communicator, &game);

    update_game(&game);

    // Gather the full game on rank 0 and print
    print_global_game(&game, rank);
  }

  // Free buffers
  destroy_communicator(&communicator);
  destroy_game(&game);

	MPI_Finalize();
  return 0;
}
