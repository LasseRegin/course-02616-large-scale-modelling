#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "update_game.h"

static inline void swap_arrays(bool** game_a, bool** game_b)
{
  bool* temp = *game_a;
  *game_a = *game_b;
  *game_b = temp;
}

static inline void print_matrix(const bool* const restrict game, const int N) {
  int N_pad = N + 2;

  for (int r = 0; r < N_pad; r++) {
    for (int c = 0; c < N_pad; c++) {
      printf("%d ", game[r * N_pad + c]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc,&argv);

	int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = 3;
  const int N_pad = N + 2;

  const bool init[5][5] = {
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 1, 1, 1, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
  };

  bool* game_a = (bool*)calloc(N_pad * N_pad, sizeof(bool));
  bool* game_b = (bool*)calloc(N_pad * N_pad, sizeof(bool));

  memcpy(game_a, init, N_pad * N_pad * sizeof(bool));

  printf("pre loop\n");
  for (int iter = 0; iter < 5; iter++) {
    print_matrix(game_a, N);

    update_game(game_a, game_b, N);
    swap_arrays(&game_a, &game_b);
  }

  // NOTE: MPI_NULL_PROC

  free(game_a);
  free(game_b);

	MPI_Finalize();
  return 0;
}
