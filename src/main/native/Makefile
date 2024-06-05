#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -std=c++11 -march=x86-64 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -mavx -fPIC
OBJS = args.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o fasttext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
opt: libfasttext.so

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext


args.o: fastText/src/args.cc fastText/src/args.h
	$(CXX) $(CXXFLAGS) -c fastText/src/args.cc

matrix.o: fastText/src/matrix.cc fastText/src/matrix.h
	$(CXX) $(CXXFLAGS) -c fastText/src/matrix.cc

dictionary.o: fastText/src/dictionary.cc fastText/src/dictionary.h fastText/src/args.h
	$(CXX) $(CXXFLAGS) -c fastText/src/dictionary.cc

loss.o: fastText/src/loss.cc fastText/src/loss.h fastText/src/matrix.h fastText/src/real.h
	$(CXX) $(CXXFLAGS) -c fastText/src/loss.cc

productquantizer.o: fastText/src/productquantizer.cc fastText/src/productquantizer.h fastText/src/utils.h
	$(CXX) $(CXXFLAGS) -c fastText/src/productquantizer.cc

densematrix.o: fastText/src/densematrix.cc fastText/src/densematrix.h fastText/src/utils.h fastText/src/matrix.h
	$(CXX) $(CXXFLAGS) -c fastText/src/densematrix.cc

quantmatrix.o: fastText/src/quantmatrix.cc fastText/src/quantmatrix.h fastText/src/utils.h fastText/src/matrix.h
	$(CXX) $(CXXFLAGS) -c fastText/src/quantmatrix.cc

vector.o: fastText/src/vector.cc fastText/src/vector.h fastText/src/utils.h
	$(CXX) $(CXXFLAGS) -c fastText/src/vector.cc

model.o: fastText/src/model.cc fastText/src/model.h
	$(CXX) $(CXXFLAGS) -c fastText/src/model.cc

utils.o: fastText/src/utils.cc fastText/src/utils.h
	$(CXX) $(CXXFLAGS) -c fastText/src/utils.cc

meter.o: fastText/src/meter.cc fastText/src/meter.h
	$(CXX) $(CXXFLAGS) -c fastText/src/meter.cc

fasttext.o: fastText/src/fasttext.cc fastText/src/*.h
	$(CXX) $(CXXFLAGS) -c fastText/src/fasttext.cc

fasttext_jni.o: $(OBJS) fasttext_jni.cc fasttext_jni.h
	$(CXX) $(CXXFLAGS) -c -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux fasttext_jni.cc -o fasttext_jni.o

libfasttext.so: $(OBJS) fasttext_jni.o
	$(CXX) -shared -fPIC -o libfasttext.so fasttext_jni.o $(OBJS) -lc

clean:
	rm -rf *.o *.gcno *.gcda
