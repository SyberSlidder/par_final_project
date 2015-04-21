

CC=nvcc
CFLAGS=-O2
OUT_NAME=cuda_out

all: main.cpp 
	$(CC) $(CFLAGS) main.cpp -o $(OUT_NAME)

run:
	./$(OUT_NAME)

clean:
	rm $(OUT_NAME)
