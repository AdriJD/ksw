#include "seatest.h"

void test_fixture_radial_functional(void);

void all_tests(void){

  test_fixture_radial_functional();
}

int main(void){

 return run_tests(all_tests);
}
