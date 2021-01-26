#include <random>
#include <iterator>
#include <algorithm>
#include <vector>
#include <iostream>
#include <omp.h>

#define MINSIZE 32768
using namespace std;

template <typename T>
void P_Merge(T *C, T * A, T *B, int na, int nb)
{
    if (na < nb) { P_Merge(C, B, A, nb, na);}
    else if (na==0) { return; }
    else // na >= nb
    {
        if(na > MINSIZE)
        {
            int ma = na/2;
            // lower_bound (first, last,val); returns an iterator pointing to the first
            // element in the range [first,last) which does not compare less than val.
            int mb = std::lower_bound(B, B+nb, A[ma]) - B;
            C[ma+mb] = A[ma];
            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    #pragma omp task
                        P_Merge(C, A, B, ma, mb);
                    #pragma omp task
                        P_Merge(C+ma+mb+1,A+ma+1,B+mb,na-ma-1,nb-mb);
                }
            }
        }
        else
        {
            std::merge(A, A+na, B, B+nb, C);
        }
    }
}

template< class Iter >
void fill_with_random_int_values( Iter start, Iter end, int min, int max)
{
    static std::random_device rd;    // you only need to initialize it once
    static std::mt19937 mte(rd());   // this is a relative big object to create

    std::uniform_int_distribution<int> dist(min, max);

    std::generate(start, end, [&] () { return dist(mte); });
}

template <typename T>
void P_MergeSort(T *B, T *A, int n)
{
    if (n<MINSIZE) {
        std::copy(A, A+n, B);
        sort(B, B+n);
    }
    else
    {
        vector<T> C(n);
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                    P_MergeSort(C.data(), A, n/2);
                #pragma omp task
                    P_MergeSort(C.data()+n/2, A+n/2, n-n/2);
            }
        }
        P_Merge(B, C.data(), C.data()+n/2, n/2, n-n/2);
    }
}

int main(int argc, char *argv[])
{
	for(int exponentbase = 14; exponentbase <= 27; exponentbase++)
    {
        int size = pow(2, exponentbase);
        std::vector<int> a(size);
        std::vector<int> b(size);

        fill_with_random_int_values(a.begin(), a.end(), 0, std::numeric_limits<int>::max());

        printf("Size: 2^%d\n", exponentbase);

        double startp = omp_get_wtime();
        P_MergeSort(b.data(), a.data(), size);
        double endp = omp_get_wtime();
        printf("P_MergeSort took %f seconds\n", endp - startp);
        
        if(std::is_sorted(b.begin(),b.end()))
        {
            cout << "Array successfully sorted with P_mergesort\n";
        }
        
    }
    
    for(int exponentbase = 14; exponentbase <= 27; exponentbase++)
    {
        int size = pow(2, exponentbase);
        std::vector<int> a(size);
        std::vector<int> b(size);
        std::vector<int> c(2*size);
        std::vector<int> d(2*size);

        fill_with_random_int_values(a.begin(), a.end(), 0, std::numeric_limits<int>::max());
        fill_with_random_int_values(b.begin(), b.end(), 0, std::numeric_limits<int>::max());
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());

        printf("Size: 2^%d\n", exponentbase);

        double startp = omp_get_wtime();
        P_Merge(c.data(), a.data(), b.data(), size, size);
        double endp = omp_get_wtime();
        printf("P_Merge took %f seconds\n", endp - startp);
        
        double starts = omp_get_wtime();
        std::merge(a.begin(), a.end(), b.begin(), b.end(), d.begin());
        double ends = omp_get_wtime();
        printf("std::merge took %f seconds\n", ends - starts);

        if(std::is_sorted(c.begin(),c.end()))
        {
            cout << "Array successfully merged with P_merge\n";
        }
        if(std::is_sorted(d.begin(),d.end()))
        {
            cout << "Array successfully merged with std::merge\n";
        }
    }
	
    return 0;
}
