CXX        = g++
CFLAGS     = -std=gnu++17 -lgfortran -Wall -O3 -w -mavx
INC        = -I faiss -I include/
LFLAGS     = faiss/build/faiss/libfaiss.a OpenBLAS/libopenblas.a -lpthread -lm -ldl -lgfortran
OMPFLAGS   = -fopenmp

ALL_CFLAGS = $(CFLAGS) $(OMPFLAGS)
ALL_LFLAGS = $(LFLAGS) $(OMPFLAGS)

index: clean_index
	$(CXX) $(INC) $(ALL_CFLAGS) \
	    src/readfile.cpp \
	    src/utils.cpp \
	    src/FilterIndexHamming.cpp \
	    src/index.cpp \
	  -o index $(ALL_LFLAGS)

query: clean_query
	$(CXX) $(INC) $(ALL_CFLAGS) \
	    src/readfile.cpp \
	    src/utils.cpp \
	    src/FilterIndexHamming.cpp \
	    src/query.cpp \
	  -o query $(ALL_LFLAGS)
	
clean_index:
	rm -f index

clean_query:
	rm -f query

.PHONY: clean all

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all
