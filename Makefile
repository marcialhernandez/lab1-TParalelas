executable:=simdsort
library:=libtest

tmp:=./tmp
src:=./src
test:=./test

#objects:=$(tmp)/entrada.o

#sources:=$(src)/entrada.cc

#cxxflags:= -g -std=c++11 -Wall
cxxflags:= -g -Wall
cxx:=g++
#thread:=-lpthread
simd:=-msse3

includes:=-I./ -I./include -I../api/include
libs:=-L./ -L./lib

main: $(objects)
	$(cxx) $(includes) $(libs) -o $(executable) $(executable).c $(cxxflags) $(simd)
	#$(cxx) $(includes) $(libs) $(objects) -o $(executable) $(executable).c $(cxxflags) $(simd)


$(tmp)/%.o: $(src)/%.c 
	test -d $(tmp) || mkdir $(tmp)
	$(cxx) $(includes) -c -o $(tmp)/$(*F).o $(src)/$*.cc $(cxxflags) $(simd)
	#$(cxx) $(includes) -c -o $(tmp)/$(*F).o $(src)/$*.cc $(cxxflags) $(simd)

testing:  $(objects)
	$(cxx) $(includes) $(libs) -o $(executable) $(executable).cc $(cxxflags) $(simd)  -DTESTING
	#$(cxx) $(includes) $(libs) $(objects) -o $(executable) $(executable).cc $(cxxflags) $(simd)  -DTESTING

testing-lib:  $(objects)
	$(cxx) $(includes) $(libs) -o $(library).so $(cxxflags) $(simd) -shared
	#$(cxx) $(includes) $(libs) $(objects) -o $(library).so $(cxxflags) $(simd) -shared


clean:
	rm -rf $(tmp);
	rm -f $(executable);

#raceTest:
	#read porConsola
	#valgrind --tool=memcheck --tool=helgrind -v --log-file=RaceConditions_test.txt ./img $ #-m test2.txt -f 2 -h 2 -p 4 