#include "seatest.h"

void test_fixture_radial_functional(void);
void test_fixture_estimator(void);
void test_fixture_fisher(void);

void all_tests(void){

  test_fixture_radial_functional();
  test_fixture_estimator();
  test_fixture_fisher();
}

int main(void){

 return run_tests(all_tests);
}
