#include <stdio.h>
#include <stdlib.h>

#include "initialize_game.h"

void print_matrix(const bool* const restrict game, const int rows, const int cols) {
  char digits[] = {'0', '1'};

  char* outdata = (char*)malloc(rows * (cols * 2 + 1) + 2 * sizeof(char));
  int outindex = 0;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      outdata[outindex++] = digits[ game[r * cols + c] ];
      outdata[outindex++] = ' ';
    }
    outdata[outindex++] = '\n';
  }
  outdata[outindex++] = '\n';
  outdata[outindex++] = '\0';

  printf("%s", outdata);
  free(outdata);
}

void print_global_game(GameInfo* game, const int rank) {
  int cols_pad = game->local_cols + 2;

  // Gather the full game on rank 0
  bool* full_game = NULL;
  if (rank == 0) {
    full_game = (bool*)calloc((game->global_rows + 2) * (game->global_cols + 2), sizeof(bool));
  }

  MPI_Gather(&game->current[cols_pad], cols_pad * game->local_rows, MPI_C_BOOL,
            &full_game[cols_pad], cols_pad * game->local_rows, MPI_C_BOOL,
            0, game->communicator);

  // rank 0 print full game and free
  if (rank == 0) {
    printf("full_game:\n");
    print_matrix(full_game, game->global_rows + 2, game->global_cols + 2);
    fflush(stdout);
    free(full_game);
  }
}
