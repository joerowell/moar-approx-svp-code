#include <fplll.h>
#include "cpuperformance.hpp"
#include <iostream>

int main() {
  // Note: the random lattice must be generated from the command line
  ZZ_mat<mpz_t> mat;
  std::cin >> mat;
  uint64_t perf_counter{};
  
  // Now we create a CPU counter.
  // N.B This will increase the value of the array we give it as a starting value.
  // Please note; this is a deliberately *newed* counter.
  // The reason why is because the rdtsc cpu counter is called in the constructor and then finalised
  // in the destructor: if we have a variable that is stack-scoped here, we won't get the right number
  // of cycles (or at all). 
  cpu::update_performance_counter* _cpuperf = new cpu::update_performance_counter(perf_counter); 
  // Here we use the wrapper: fplll will decide on the right type of precision and so on.
  // This is likely to account for seemingly random spikes in the cost: it might be wise to fit
  // a disjointed curve.
  lll_reduction(mat);
  delete _cpuperf;
  std::cout << mat.get_rows() << "," << perf_counter << std::endl;
}
