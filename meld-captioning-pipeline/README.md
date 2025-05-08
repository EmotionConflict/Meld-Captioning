# MELD Captioning Pipeline

Generate multimodal captions (visual, audio, text) for MELD dataset.

# How to Build OpenFace on macOS
Run: 
brew install cmake boost dlib opencv@3 tbb git wget
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash ./download_models.sh

mkdir build
cd build
brew install openblas dlib
## if it cannot find, cmake -D OpenCV_DIR=$(brew --prefix opencv@3)/share/OpenCV ..
cmake -D CMAKE_BUILD_TYPE=RELEASE .. 
make -j$(sysctl -n hw.ncpu)

## confirm if worked
ls ./bin/FeatureExtraction 

## Run on simple example
./bin/FeatureExtraction -f ../samples/default.wmv -out_dir ../output

## If not work, fix using below inside build folder
rm -rf *
## Now at root directory OpenFace
find . -name CMakeLists.txt -exec sed -i '' '/stdc++fs/d' {} +
grep -r "stdc++fs" .
## This should not return anything
## Now rerun, in build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(sysctl -n hw.ncpu)

## Final check
ls ./bin/FeatureExtraction


