#pragma once

#include <mpi.h>
#include <stdbool.h>

typedef struct GameInfo {
  int global_rows;
  int global_cols;
  int local_rows;
  int local_cols;

  bool* restrict current;
  bool* restrict previouse;

  MPI_Comm communicator;
} GameInfo;

int initialize_game(GameInfo* game, const int size, const int rows, const int cols, const bool* const restrict init);
void destroy_game(GameInfo* game);
