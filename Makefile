
gameoflife: main.o update_game.o
	mpicc main.o update_game.o -o gameoflife

clean:
	rm -f *.o
	rm -f gameoflife

%.o: %.c
	mpicc -std=gnu99 -c -o $@ $<
