#include <stdfloat>
#include <iostream>
#include <type_traits>

int main() {
    #ifdef __STDCPP_BFLOAT16_T__
    std::cout << "std::bfloat16_t is supported!" << std::endl;
    std::cout << "Size of bfloat16_t: " << sizeof(std::bfloat16_t) << " bytes" << std::endl;
    
    std::bfloat16_t x = 1.5bf16;
    std::cout << "Test value: " << static_cast<float>(x) << std::endl;
    return 0;
    #else
    std::cout << "std::bfloat16_t is NOT supported" << std::endl;
    return 1;
    #endif
}
