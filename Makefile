CC=nvcc
CFLAGS=-O0
OUT_NAME=cuda_out
FILES = main.cu \
	sgemm_v1.cu \
	squareSumVector.cu \
	maxwell_sgemm.cu

all: $(FILES) 
	$(CC) $(CFLAGS) $(FILES) -lcublas -o $(OUT_NAME)

run:
	./$(OUT_NAME)

clean:
	rm $(OUT_NAME)
