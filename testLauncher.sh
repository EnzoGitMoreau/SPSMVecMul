#!/bin/zsh

# Loop through numbers 1 to 8
for i in {1..8}
do
  # Run the program with the current number as an argument
  ./tests $i $((2*4*5*7*3 * 2 )) 

done
