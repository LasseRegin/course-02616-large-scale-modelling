
OBJS = main.o initialize_game.o synchronize_game.o update_game.o debug_game.o

all: gameoflife

gameoflife: $(OBJS)
	mpicc $(OBJS) -o gameoflife

clean:
	rm -f *.o
	rm -f gameoflife

%.o: %.c
	mpicc -std=gnu99 -c -o $@ $<
