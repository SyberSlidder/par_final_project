

CC=nvcc
CFLAGS=-O0
OUT_NAME=cuda_out
FILES = main.cpp \
	sgemm_v1.cu \
	squareSumVector.cu

all: $(FILES) 
	$(CC) $(CFLAGS) $(FILES) -o $(OUT_NAME)

run:
	./$(OUT_NAME)

clean:
	rm $(OUT_NAME)
