#pragma once

#include <mpi.h>

#include "initialize_game.h"

typedef struct GameCommunicator {
  int north_rank;
  int south_rank;

  bool* restrict north_recv;
  bool* restrict south_recv;

  MPI_Request* restrict request;
  MPI_Status* restrict status;
} GameCommunicator;

void initialize_communicator(GameCommunicator* communicator, GameInfo* game, const int size, const int rank);
void synchronize_game(GameCommunicator* communicator, GameInfo* game);
void destroy_communicator(GameCommunicator* communicator);
