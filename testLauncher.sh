#!/bin/zsh

# Loop through numbers 1 to 8
for i in {1..8}
do
  # Run the program with the current number as an argument
  ./tests $i
done
