
#include <mpi.h>
#include <stdlib.h>
#include <stdbool.h>

#include "initialize_game.h"

int initialize_game(GameInfo* game, const int size, int rows, int cols, const bool* const restrict init)
{
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Ensure that the data can be spread evenly
  if (rows % size != 0) {
    return 1;
  }

  // Set global properties
  game->global_rows = rows;
  game->global_cols = cols;

  // Set communicator
  game->communicator = MPI_COMM_WORLD;

  // Calculate the number of rows each process is responsibol for
  game->local_rows = rows / size;
  game->local_cols = cols;

  int local_cols_pad = game->local_cols + 2;
  int local_rows_pad = game->local_rows + 2;

  // Allocate buffers
  game->current = (bool*)calloc(local_rows_pad * local_cols_pad, sizeof(bool));
  game->previouse = (bool*)calloc(local_rows_pad * local_cols_pad, sizeof(bool));

  // Scatter initial data. This is just for distributing the loaded data,
  // not for handling boundery conditions.
  bool* init_partial = (bool*)calloc(game->local_rows * game->local_cols, sizeof(bool));
  MPI_Scatter(init        , game->local_rows * game->local_cols, MPI_C_BOOL,
              init_partial, game->local_rows * game->local_cols, MPI_C_BOOL,
              0, game->communicator);

  // Move `init_partial` into the game buffer where the boundaries are considered
  for (int init_row = 0; init_row < game->local_rows; init_row++) {
    int game_row = init_row + 1;

    for (int init_col = 0; init_col < game->local_cols; init_col++) {
      int game_col = init_col + 1;

      game->current[game_row * local_cols_pad + game_col] = init_partial[init_row * game->local_cols + init_col];
    }
  }
  free(init_partial);

  return 0;
}

void destroy_game(GameInfo* game) {
  free(game->current);
  free(game->previouse);
}
