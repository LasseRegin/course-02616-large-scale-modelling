
#include <stdio.h>
#include <stdbool.h>

#include "initialize_game.h"
#include "update_game.h"

static inline void swap_arrays(bool* restrict * array_a, bool* restrict * array_b)
{
  bool* temp = *array_a;
  *array_a = *array_b;
  *array_b = temp;
}

void update_game(GameInfo* game)
{
  int cols_pad = game->local_cols + 2;
  bool* current = game->current;

  for (int r = 1; r < game->local_rows + 1; r++) {
    for (int c = 1; c < game->local_cols + 1; c++) {
      int sum = current[(r - 1) * cols_pad + c - 1] + current[(r - 1) * cols_pad + c + 0] + current[(r - 1) * cols_pad + c + 1]
              + current[(r + 0) * cols_pad + c - 1] +                  0                  + current[(r + 0) * cols_pad + c + 1]
              + current[(r + 1) * cols_pad + c - 1] + current[(r + 1) * cols_pad + c + 0] + current[(r + 1) * cols_pad + c + 1];

      // Update previouse array, later previouse and current will be swapped
      if (sum == 3 || (sum == 2 && current[r * cols_pad + c] == 1)) {
        game->previouse[r * cols_pad + c] = 1;
      } else {
        game->previouse[r * cols_pad + c] = 0;
      }
    }
  }

  // Now make previouse the current
  swap_arrays(&game->current, &game->previouse);
}
