CC=gcc
CFLAGS= -std=c99 -O3 -fPIC -Wall
SOURCES=basis_functions.c
LIB=libttv2fast2furious.so
OBJECTS=$(SOURCES:.c=.o)

all: $(SOURCES) libttv2fast2furious.so

libttv2fast2furious.so: $(OBJECTS)
	@echo ""
	@echo "Linking shared library $@ ..."
	$(CC) $(CFLAGS) -lm -lgsl -lgslcblas -shared $(OBJECTS) -o $@
	@echo ""
	@echo "The shared library $< has been created successfully."

%.o: %.c
	@echo "Compiling source file $< ..."
	$(CC)  -c -o $@ $< $(CFLAGS)

clean:
	@echo "Removing object files *.o ..."
	@-rm -f *.o
	@echo "Removing shared library libcelmech.so ..."
	@-rm -f *.so
