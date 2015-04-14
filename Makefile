executable:=simdsort
library:=libtest

tmp:=./tmp
src:=./src
test:=./test

#objects:=$(tmp)/entrada.o

#sources:=$(src)/entrada.cc

#cxxflags:= -g -std=c++11 -Wall
cxxflags:= -g -std=c++11 -Wall
cxx:=g++
#thread:=-lpthread
simd:=-msse3

includes:=-I./ -I./include -I../api/include
libs:=-L./ -L./lib

main: $(objects)
	$(cxx) $(includes) $(libs) -o $(executable) $(executable).c $(cxxflags) $(simd)

$(tmp)/%.o: $(src)/%.c 
	test -d $(tmp) || mkdir $(tmp)
	$(cxx) $(includes) -c -o $(tmp)/$(*F).o $(src)/$*.c $(cxxflags) $(simd)

testing:  $(objects)
	$(cxx) $(includes) $(libs) -o $(executable) $(executable).c $(cxxflags) $(simd)  -DTESTING

testing-lib:  $(objects)
	$(cxx) $(includes) $(libs) -o $(library).so $(cxxflags) $(simd) -shared

clean:
	rm -rf $(tmp);
	rm -f $(executable);