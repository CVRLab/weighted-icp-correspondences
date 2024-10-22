mkdir -p build
cd build
cmake ..
make
cd ..
mv build/alignment_experiments ./
