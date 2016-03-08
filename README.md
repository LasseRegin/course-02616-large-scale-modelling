# Game of life

> Cluster parallelization of game of life using MPI. This was developed during
the DTU course "02616 Large-scale Modelling". If you are a participant of this
course you are not allowed to use it.

## Compile and execute

```shell
make
mpiexec -n $PROCESSES gameoflife
```

Note that `$PROCESSES` must divide the number of cells in the game-of-life
initialization matrix defined in `main.c`.
