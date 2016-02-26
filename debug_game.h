#pragma once

#include "initialize_game.h"

void print_matrix(const bool* const restrict game, const int rows, const int cols);
void print_global_game(GameInfo* game, const int rank);
