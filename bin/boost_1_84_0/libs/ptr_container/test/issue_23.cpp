// Use, modification and distribution is subject to the
// Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
 
#include <boost/ptr_container/ptr_vector.hpp>

template<class T> struct Foo
{
    static void test(boost::ptr_vector<Foo<T> >& /*t*/)
    {
    }
};

int main()
{
    boost::ptr_vector<Foo<double> > ptr;
    return 0;
}
