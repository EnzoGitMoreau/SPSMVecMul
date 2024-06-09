<h1>What is this project about ?</h1>
<br>
Building a faster way of doing Matrix-Vector Multiplication in the case of SparseSymmetric Matrices.

SparseSymmetric Matrices are mostly used in dynamic and mechanical systems. We use a lot of those in our software Cytosim at Sainsbury's Laboratory.

Using a new algorithmic approach, we managed to fasten the computation in a multi-threaded way, increasing the performance.

You can have a look at the publish paper here:

*Futur link of paper*
<br>
<h2>How to use
</h2>
Please download this repository, the algorithm does not require any additional library and is built using C++20 only. If you can't use C++20, you can chose the BOOST/OpenMP branches that will allow you to use multi-threading.

It is an header-only library, very easy to use and implement thanks to the BLAS specification 

*Link to BLAS*