#!/bin/zsh

# Loop through numbers 1 to 8

brew install openblas
brew install openmp
brew install llvm

wget https://developer.arm.com/-/media/Files/downloads/hpc/arm-performance-libraries/24-04/macos/arm-performance-libraries_24.04_macOS.tgz?rev=17145f380ea64fe6b26de596182561bc&revision=17145f38-0ea6-4fe6-b26d-e596182561bc
chmod -x ampl_24.04_flang-new_clang_18_install.sh
sudo ./ampl_24.04_flang-new_clang_18_install.sh -y