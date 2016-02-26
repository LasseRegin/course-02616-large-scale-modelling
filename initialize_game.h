#pragma once

#include <mpi.h>
#include <stdbool.h>

struct TopologyDirection {
  int rank;
  bool* restrict recv;
  MPI_Datatype send_type;
  MPI_Datatype send_type_vec;
};

struct Topology {
  struct TopologyDirection north;
  struct TopologyDirection north_west;
  struct TopologyDirection north_east;

  struct TopologyDirection south;
  struct TopologyDirection south_west;
  struct TopologyDirection south_east;

  struct TopologyDirection east;
  struct TopologyDirection west;
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

  // Topology information and buffers
  struct Topology topology;

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
