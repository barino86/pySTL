# pySTL
Full Python Implementation of the R stl package 
including porting the original fortran code to 
python. 

I kept the 1-indexing to get it working correctly, 
and will eventually update it to use python's 
0-indexing.

There is still a lot of work that needs to be done, 
but for all intents and purposes, the algorithm
works and gives almost identical results to the 
original fortran code. Numbers might be off in the 
8-10 decimal due to rounding.
