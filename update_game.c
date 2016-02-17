
#include <stdio.h>
#include <stdbool.h>

#include "update_game.h"

void update_game(const bool * const restrict in_game, bool * const restrict out_game, const int N)
{
  int N_pad = N + 2;

  for (int r = 1; r < N + 1; r++) {
    for (int c = 1; c < N + 1; c++) {
      int sum = in_game[(r - 1) * N_pad + c - 1] + in_game[(r - 1) * N_pad + c + 0] + in_game[(r - 1) * N_pad + c + 1]
              + in_game[(r + 0) * N_pad + c - 1] +                0                 + in_game[(r + 0) * N_pad + c + 1]
              + in_game[(r + 1) * N_pad + c - 1] + in_game[(r + 1) * N_pad + c + 0] + in_game[(r + 1) * N_pad + c + 1];

      if (sum == 3 || (sum == 2 && in_game[r * N_pad + c] == 1)) {
        out_game[r * N_pad + c] = 1;
      } else {
        out_game[r * N_pad + c] = 0;
      }
    }
  }
}
