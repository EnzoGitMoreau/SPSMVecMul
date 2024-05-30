#!/bin/zsh

# Loop through numbers 1 to 8
rm -f res/*.csv
for i in {1..10}
do
  # Run the program with the current number as an argument
  ./tests 10 10000 100 $(( $i*0.1)) 


done

