
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "initialize_game.h"
#include "synchronize_game.h"

static const int SEND_NORTH_TAG = 10;
static const int SEND_SOUTH_TAG = 11;

void initialize_communicator(GameCommunicator* communicator, GameInfo* game, const int size, const int rank) {
  // Define north and south ranks
  communicator->north_rank = rank - 1;
  communicator->south_rank = rank + 1;
  if (rank == 0) communicator->north_rank = MPI_PROC_NULL;
  if (rank == size - 1) communicator->south_rank = MPI_PROC_NULL;

  // Allocate boundary communication buffers
  communicator->north_recv = (bool*)calloc(game->local_cols, sizeof(bool));
  communicator->south_recv = (bool*)calloc(game->local_cols, sizeof(bool));

  // Allocate holders for request objectss
  communicator->request = (MPI_Request*) malloc(4 * sizeof(MPI_Request));
  communicator->status = (MPI_Status*) malloc(4 * sizeof(MPI_Status));
}

void synchronize_game(GameCommunicator* communicator, GameInfo* game) {
  int cols_pad = game->local_cols + 2;

  // send data to north, receive from south
  MPI_Isend(&game->current[cols_pad + 1], game->local_cols, MPI_C_BOOL,
            communicator->north_rank, SEND_NORTH_TAG, game->communicator,
            &communicator->request[0]);

  MPI_Irecv(communicator->south_recv, game->local_cols, MPI_C_BOOL,
            communicator->south_rank, SEND_NORTH_TAG, game->communicator,
            &communicator->request[1]);

  // send data to south, receive from north
  MPI_Isend(&game->current[game->local_rows * cols_pad + 1], game->local_cols, MPI_C_BOOL,
            communicator->south_rank, SEND_SOUTH_TAG, game->communicator,
            &communicator->request[2]);

  MPI_Irecv(communicator->north_recv, game->local_cols, MPI_C_BOOL,
            communicator->north_rank, SEND_SOUTH_TAG, game->communicator,
            &communicator->request[3]);

  // await completion
  MPI_Waitall(4, communicator->request, communicator->status);

  // transfer receive buffers to game buffers
  memcpy(&game->current[1], communicator->north_recv,
         game->local_cols * sizeof(bool));
  memcpy(&game->current[(game->local_rows + 1) * cols_pad + 1], communicator->south_recv,
         game->local_cols * sizeof(bool));
}

void destroy_communicator(GameCommunicator* communicator) {
  free(communicator->north_recv);
  free(communicator->south_recv);
  free(communicator->request);
  free(communicator->status);
}
