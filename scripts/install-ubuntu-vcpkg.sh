#!/bin/bash

set -xeu

# Remove unused files to increase available disk space
rm -rf /usr/local/.ghcup
rm -rf /opt/hostedtoolcache/CodeQL
rm -rf /usr/local/lib/android/sdk/ndk
rm -rf /usr/share/dotnet
rm -rf /opt/ghc
rm -rf /usr/local/share/boost

apt-get update
apt-get install -y \
	autoconf \
	autoconf-archive \
	automake \
	bison \
	build-essential \
	clang \
	cmake \
	curl \
	git \
	gperf \
	libclang-dev \
	libdbus-1-dev \
	libgles2-mesa-dev \
	libharfbuzz0b \
	libltdl-dev \
	libtool \
	libx11-dev \
	libxcursor-dev \
	libxdamage-dev \
	libxext-dev \
	libxft-dev \
	libxi-dev \
	libxinerama-dev \
	libxrandr-dev \
	libxtst-dev \
	python3-jinja2 \
	sudo \
	tar \
	unzip \
	zip

pushd . > /dev/null
# Download and extract it
curl -OL https://ftp.gnu.org/gnu/autoconf/autoconf-2.71.tar.xz
tar -xf autoconf-2.71.tar.xz

# Configure, make, and install it
cd autoconf-2.71/
./configure
make 
sudo make install

popd > /dev/null
# Re-source your profile to bring in the updates
. ~/.profile

export VCPKG_DEFAULT_TRIPLET=x64-linux

script_dir="$(dirname "$(readlink -f "$BASH_SOURCE")")"

. "$script_dir/install-vcpkg.sh"