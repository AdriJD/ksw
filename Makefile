.PHONY: clean all check python

DIR := ${CURDIR}

SDIR = $(DIR)/src
LDIR = $(DIR)/lib
ODIR = $(DIR)/obj
IDIR = $(DIR)/include
TDIR = $(DIR)/tests
CDIR = $(DIR)/cython
PDIR = $(DIR)/ksw

HS_SDIR = $(DIR)/hyperspherical/src
HS_IDIR = $(DIR)/hyperspherical/include

NEWDIRS = $(LDIR) $(ODIR) $(TDIR)/obj $(TDIR)/bin
$(info $(shell mkdir -p -v $(NEWDIRS)))

CFLAGS = -g -Wall -fpic -std=c99
OMPFLAG = -fopenmp
OPTFLAG = -march=native -O3 -ffast-math

OBJECTS = $(ODIR)/radial_functional.o \
          $(ODIR)/common.o \
          $(ODIR)/hyperspherical.o

TEST_OBJECTS = $(TDIR)/obj/seatest.o \
               $(TDIR)/obj/test_radial_functional.o \
               $(TDIR)/obj/run_tests.o

all: $(LDIR)/libradial_functional.so 

python: $(LDIR)/libradial_functional.so setup.py $(CDIR)/radial_functional.pyx $(CDIR)/radial_functional.pxd
	python setup.py build_ext --inplace

$(LDIR)/libradial_functional.so: $(OBJECTS)
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -shared -o $(LDIR)/libradial_functional.so $(OBJECTS)

$(ODIR)/radial_functional.o: $(SDIR)/radial_functional.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(HS_IDIR)

$(ODIR)/common.o: $(HS_SDIR)/common.c $(HS_IDIR)/common.h $(HS_IDIR)/svnversion.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(HS_IDIR)

$(ODIR)/hyperspherical.o: $(HS_SDIR)/hyperspherical.c $(HS_IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(HS_IDIR)

check: check_c check_python

check_c: $(TDIR)/bin/run_tests
	$(TDIR)/bin/run_tests

check_python:
	cd $(TDIR); python -m pytest python/

$(TDIR)/bin/run_tests: $(TEST_OBJECTS) $(OBJECTS) $(LDIR)/libradial_functional.so
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -o $@ $(TEST_OBJECTS) -I$(IDIR) -I$(HS_IDIR) -I$(TDIR)/include -L$(LDIR) -lradial_functional -lm -lgomp -Wl,-rpath,$(LDIR)

$(TDIR)/obj/run_tests.o: $(TDIR)/src/run_tests.c $(TDIR)/include/seatest.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(TDIR)/include -I$(IDIR)

$(TDIR)/obj/seatest.o: $(TDIR)/src/seatest.c $(TDIR)/include/seatest.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(TDIR)/include

$(TDIR)/obj/test_radial_functional.o: $(TDIR)/src/test_radial_functional.c $(IDIR)/*.h
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPTFLAG) -c -o $@ $< -I$(IDIR) -I$(HS_IDIR) -I$(TDIR)/include

clean:
	rm -rf $(ODIR)
	rm -rf $(LDIR)
	rm -rf $(TDIR)/obj
	rm -rf $(TDIR)/bin
	rm -f $(CDIR)/*.c
	rm -f $(PDIR)/*.so
	rm -rf $(DIR)/build
	rm -rf $(PDIR)/__pycache__
	rm -rf $(TDIR)/python/__pycache__
	rm -rf $(DIR)/*.egg-info
