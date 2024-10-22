apt-get update
apt-get install -y \
  build-essential \
  cmake \
  software-properties-common

# Install OpenCV
apt-get install -y libopencv-dev python3-opencv
ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2

# Install MRPT
add-apt-repository -y ppa:joseluisblancoc/mrpt-stable
apt-get install -y libmrpt-dev mrpt-apps
apt-get install -y python3-pymrpt

# Install Point Cloud Library
apt install libpcl-dev
cd /usr/include
ln -s /usr/include/pcl-1.12/pcl /usr/include/pcl
