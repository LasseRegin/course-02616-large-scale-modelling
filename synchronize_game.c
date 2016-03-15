
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "initialize_game.h"
#include "synchronize_game.h"

static const int SEND_NORTH_TAG = 10;
static const int SEND_NORTH_WEST_TAG = 11;
static const int SEND_NORTH_EAST_TAG = 12;

static const int SEND_SOUTH_TAG = 13;
static const int SEND_SOUTH_WEST_TAG = 14;
static const int SEND_SOUTH_EAST_TAG = 15;

static const int SEND_EAST_TAG = 16;
static const int SEND_WEST_TAG = 17;

static inline void synchronize_direction(
  const GameInfo* const game,
  const struct TopologyDirection* const send,
  const struct TopologyDirection* const recv,
  MPI_Request* const restrict request_array,
  const int tag)
{
  MPI_Isend(&game->current[send->send_offset], 1, send->send_type,
            send->rank, tag, game->communicator, &request_array[0]);

  MPI_Irecv(&game->current[recv->recv_offset], 1, recv->recv_type,
            recv->rank, tag, game->communicator, &request_array[1]);
}
static inline void synchronize_direction_new(
  const GameInfo* const game,
  const struct TopologyDirectionNew* const send,
  const struct TopologyDirectionNew* const recv,
  MPI_Request* const restrict request_array,
  const int tag)
{
  MPI_Isend(game->current, 1, send->send_type,
            send->rank, tag, game->communicator, &request_array[0]);

  MPI_Irecv(game->current, 1, recv->recv_type,
            recv->rank, tag, game->communicator, &request_array[1]);
}

void synchronize_game(const GameInfo* const game)
{
  // north -> south
  synchronize_direction_new(game,
                        &game->topology.north, &game->topology.south,
                        &game->request[0], SEND_NORTH_TAG);
  // north west -> south east
  synchronize_direction(game,
                        &game->topology.north_west, &game->topology.south_east,
                        &game->request[2], SEND_NORTH_WEST_TAG);
  // north east -> south west
  synchronize_direction(game,
                        &game->topology.north_east, &game->topology.south_west,
                        &game->request[4], SEND_NORTH_EAST_TAG);

  // south -> north
  synchronize_direction_new(game,
                        &game->topology.south, &game->topology.north,
                        &game->request[6], SEND_SOUTH_TAG);
  // south west -> north east
  synchronize_direction(game,
                        &game->topology.south_west, &game->topology.north_east,
                        &game->request[8], SEND_SOUTH_WEST_TAG);
  // south east -> north west
  synchronize_direction(game,
                        &game->topology.south_east, &game->topology.north_west,
                        &game->request[10], SEND_SOUTH_EAST_TAG);

  // east -> west
  synchronize_direction(game,
                        &game->topology.east, &game->topology.west,
                        &game->request[12], SEND_EAST_TAG);
  // west -> east
  synchronize_direction(game,
                        &game->topology.west, &game->topology.east,
                        &game->request[14], SEND_WEST_TAG);

  MPI_Waitall(16, game->request, game->status);
}
