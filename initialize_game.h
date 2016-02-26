#pragma once

#include <mpi.h>
#include <stdbool.h>

struct TopologyRanks {
  int north;
  int north_west;
  int north_east;

  int south;
  int south_west;
  int south_east;

  int east;
  int west;
};

struct TopologyRecv {
  bool* restrict north;
  bool* restrict north_west;
  bool* restrict north_east;

  bool* restrict south;
  bool* restrict south_west;
  bool* restrict south_east;

  bool* restrict east;
  bool* restrict west;
};

typedef struct GameInfo {
  // size holders
  int node_dims[2];
  int global_rows;
  int global_cols;
  int local_rows;
  int local_cols;

  // game data holders
  bool* restrict current;
  bool* restrict previouse;

  // rank holders
  struct TopologyRanks rank;

  // receive buffers
  struct TopologyRecv recv;

  // Request and status buffers for MPI
  MPI_Request* restrict request;
  MPI_Status* restrict status;

  // cartesian communicator
  MPI_Comm communicator;
} GameInfo;

int initialize_game(
  GameInfo* const game,
  const int size, const int rank,
  int rows, int cols, const bool* const restrict init);

void destroy_game(GameInfo* const game);
