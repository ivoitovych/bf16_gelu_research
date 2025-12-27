/**
 * @file ulp_calculator.cpp
 * @brief Reference ULP (Unit in the Last Place) Calculator for bfloat16
 *
 * This file implements a comprehensive ULP calculator for bfloat16 floating-point
 * values. ULP is a measure of the spacing between floating-point numbers and is
 * commonly used to assess the accuracy of numerical algorithms.
 *
 * ## What is ULP?
 *
 * ULP (Unit in the Last Place) represents the distance between two adjacent
 * floating-point numbers. For error analysis, we count how many representable
 * floating-point numbers exist between two values - this count is the ULP error.
 *
 * ## bfloat16 Format
 *
 * bfloat16 (Brain Floating Point 16) uses 16 bits:
 *   - 1 bit  : Sign
 *   - 8 bits : Exponent (same as float32)
 *   - 7 bits : Mantissa (truncated from float32's 23 bits)
 *
 * This format was designed for machine learning where the dynamic range matters
 * more than precision. It can represent the same range as float32 but with
 * reduced precision.
 *
 * ## Implementation Strategy
 *
 * 1. Enumerate all 65536 possible 16-bit patterns
 * 2. Convert each pattern to bfloat16 using std::memcpy (avoiding UB)
 * 3. Filter out non-numeric values (NaN, Inf)
 * 4. Sort valid values in ascending numerical order
 * 5. Assign sequential ULP indices, with equal values sharing the same index
 * 6. Build a lookup table: bit_pattern -> ULP_index
 *
 * The ULP error between two bfloat16 values is then simply:
 *   |ulp_index[bits_of_a] - ulp_index[bits_of_b]|
 *
 * @author Claude Code
 * @date 2024
 */

#include <stdfloat>      // For std::bfloat16_t (C++23)
#include <cstdint>       // For uint16_t, int64_t
#include <cstring>       // For std::memcpy
#include <cmath>         // For std::isnan, std::isinf
#include <vector>        // For std::vector
#include <algorithm>     // For std::sort
#include <array>         // For std::array
#include <iostream>      // For std::cout
#include <iomanip>       // For std::setprecision
#include <cassert>       // For assert
#include <limits>        // For std::numeric_limits

/**
 * @brief Compile-time verification of type sizes
 *
 * We use static_assert to ensure our assumptions about type sizes are correct.
 * This prevents silent failures on platforms with unexpected type sizes.
 */
static_assert(sizeof(std::bfloat16_t) == 2, "bfloat16_t must be 2 bytes");
static_assert(sizeof(uint16_t) == 2, "uint16_t must be 2 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes for internal calculations");
static_assert(sizeof(double) == 8, "double must be 8 bytes for reference calculations");

/**
 * @brief Convert bfloat16 to its underlying uint16 bit representation
 *
 * We use std::memcpy for type punning instead of reinterpret_cast or union
 * tricks. This is the ONLY well-defined way to perform type punning in C++.
 *
 * Why not reinterpret_cast?
 *   reinterpret_cast between unrelated types violates strict aliasing rules
 *   and results in undefined behavior. The compiler may optimize incorrectly.
 *
 * Why not union?
 *   Reading from a union member other than the one written to is undefined
 *   behavior in C++ (though defined in C). This is a common source of bugs.
 *
 * Why std::memcpy?
 *   std::memcpy is explicitly defined to copy the object representation.
 *   Modern compilers recognize this pattern and optimize it to a no-op or
 *   a simple register move - there's no actual memory copy at runtime.
 *
 * @param value The bfloat16 value to convert
 * @return The 16-bit unsigned integer with the same bit pattern
 */
inline uint16_t bfloat16_to_bits(std::bfloat16_t value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/**
 * @brief Convert uint16 bit pattern to bfloat16
 *
 * This is the inverse of bfloat16_to_bits. We use std::memcpy for the same
 * reasons outlined above - it's the only well-defined way to perform this
 * type punning operation.
 *
 * @param bits The 16-bit pattern to interpret as bfloat16
 * @return The bfloat16 value with the same bit pattern
 */
inline std::bfloat16_t bits_to_bfloat16(uint16_t bits) {
    std::bfloat16_t value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * @brief Check if a bfloat16 value is a valid number (not NaN or Inf)
 *
 * For ULP calculations, we only consider finite numeric values.
 * NaN (Not a Number) and Inf (Infinity) are excluded because:
 *   - NaN is not ordered (NaN != NaN)
 *   - Infinity represents overflow, not a specific value
 *   - The concept of "distance" doesn't apply meaningfully to these
 *
 * bfloat16 special value encoding:
 *   - Exponent = 0xFF (all ones), Mantissa = 0: Infinity
 *   - Exponent = 0xFF (all ones), Mantissa != 0: NaN
 *
 * @param value The bfloat16 value to check
 * @return true if the value is finite (not NaN and not Inf)
 */
inline bool is_finite_bfloat16(std::bfloat16_t value) {
    // Convert to float for standard library function compatibility
    // This conversion is exact for all finite bfloat16 values
    float f = static_cast<float>(value);
    return !std::isnan(f) && !std::isinf(f);
}

/**
 * @brief Structure to hold a bfloat16 value with its bit representation
 *
 * We store both the bfloat16 value and its bit pattern together to avoid
 * repeated conversions during sorting and index assignment.
 */
struct Bfloat16Entry {
    std::bfloat16_t value;  ///< The bfloat16 floating-point value
    uint16_t bits;          ///< The underlying 16-bit representation

    /**
     * @brief Comparison operator for sorting in ascending order
     *
     * We compare the float representations to get proper numerical ordering.
     * This correctly handles:
     *   - Negative numbers (which have the sign bit set)
     *   - Subnormal numbers (very small values near zero)
     *   - The relationship between -0 and +0
     */
    bool operator<(const Bfloat16Entry& other) const {
        return static_cast<float>(value) < static_cast<float>(other.value);
    }
};

/**
 * @class UlpCalculator
 * @brief Reference ULP calculator for bfloat16 floating-point values
 *
 * This class builds a complete lookup table mapping every valid bfloat16
 * bit pattern to its ULP index. The ULP index represents the position of
 * a value in the sorted sequence of all representable bfloat16 numbers.
 *
 * ## Usage Example
 *
 * ```cpp
 * UlpCalculator calc;
 *
 * std::bfloat16_t a = 1.0bf16;
 * std::bfloat16_t b = 1.0078125bf16;  // Next representable value after 1.0
 *
 * int64_t error = calc.ulp_distance(a, b);  // Returns 1
 * ```
 *
 * ## Design Notes
 *
 * - The lookup table uses int64_t to allow for signed differences
 * - Invalid entries (NaN, Inf) are marked with -1
 * - Equal values (like +0 and -0) share the same index
 * - The table is built once at construction and reused for all queries
 */
class UlpCalculator {
public:
    /// Marker value for invalid (non-numeric) bit patterns in the lookup table
    static constexpr int64_t INVALID_ULP_INDEX = -1;

    /// Total number of possible 16-bit patterns
    static constexpr size_t TOTAL_PATTERNS = 65536;

private:
    /**
     * @brief Lookup table mapping bit patterns to ULP indices
     *
     * ulp_index_table[bits] gives the ULP index for the bfloat16 value
     * with bit representation 'bits'.
     *
     * - Valid indices are non-negative integers
     * - INVALID_ULP_INDEX (-1) indicates NaN or Inf values
     */
    std::array<int64_t, TOTAL_PATTERNS> ulp_index_table;

    /// Number of valid (finite) bfloat16 values
    size_t valid_count;

public:
    /**
     * @brief Construct the ULP calculator and build the lookup table
     *
     * Construction involves:
     * 1. Enumerating all 65536 bit patterns
     * 2. Converting each to bfloat16 and filtering non-numeric values
     * 3. Sorting valid values in ascending order
     * 4. Assigning ULP indices (equal values share the same index)
     * 5. Populating the lookup table
     *
     * Time complexity: O(n log n) where n = 65536
     * Space complexity: O(n) for the sorted vector and lookup table
     */
    UlpCalculator() {
        build_ulp_table();
    }

    /**
     * @brief Get the ULP distance between two bfloat16 values
     *
     * The ULP distance is the absolute difference between the ULP indices
     * of two values. This represents how many representable floating-point
     * numbers exist between them (inclusive of one endpoint).
     *
     * @param a First bfloat16 value
     * @param b Second bfloat16 value
     * @return The ULP distance, or -1 if either value is non-numeric
     */
    int64_t ulp_distance(std::bfloat16_t a, std::bfloat16_t b) const {
        int64_t idx_a = get_ulp_index(a);
        int64_t idx_b = get_ulp_index(b);

        // Return -1 if either value is invalid (NaN or Inf)
        if (idx_a == INVALID_ULP_INDEX || idx_b == INVALID_ULP_INDEX) {
            return INVALID_ULP_INDEX;
        }

        // Return absolute difference
        return std::abs(idx_a - idx_b);
    }

    /**
     * @brief Get the ULP index for a bfloat16 value
     *
     * @param value The bfloat16 value to look up
     * @return The ULP index, or INVALID_ULP_INDEX for NaN/Inf
     */
    int64_t get_ulp_index(std::bfloat16_t value) const {
        uint16_t bits = bfloat16_to_bits(value);
        return ulp_index_table[bits];
    }

    /**
     * @brief Get the ULP index for a raw bit pattern
     *
     * @param bits The 16-bit pattern to look up
     * @return The ULP index, or INVALID_ULP_INDEX for NaN/Inf patterns
     */
    int64_t get_ulp_index_from_bits(uint16_t bits) const {
        return ulp_index_table[bits];
    }

    /**
     * @brief Get the number of valid (finite) bfloat16 values
     *
     * This excludes all NaN and Infinity representations.
     *
     * @return Count of valid bfloat16 values
     */
    size_t get_valid_count() const {
        return valid_count;
    }

    /**
     * @brief Get the maximum ULP index (number of distinct values - 1)
     *
     * Note: This may be less than valid_count - 1 because +0 and -0
     * share the same index.
     *
     * @return The highest ULP index assigned
     */
    int64_t get_max_ulp_index() const {
        int64_t max_idx = 0;
        for (size_t i = 0; i < TOTAL_PATTERNS; ++i) {
            if (ulp_index_table[i] > max_idx) {
                max_idx = ulp_index_table[i];
            }
        }
        return max_idx;
    }

private:
    /**
     * @brief Build the ULP index lookup table
     *
     * This is the core algorithm that creates the mapping from bit patterns
     * to ULP indices. The process is:
     *
     * 1. Initialize all entries to INVALID_ULP_INDEX
     * 2. Collect all valid (finite) bfloat16 values with their bit patterns
     * 3. Sort them by numerical value
     * 4. Assign ULP indices, keeping equal values at the same index
     * 5. Store indices in the lookup table
     */
    void build_ulp_table() {
        // Step 1: Initialize all entries as invalid
        // This ensures NaN and Inf patterns remain marked as invalid
        ulp_index_table.fill(INVALID_ULP_INDEX);

        // Step 2: Collect all valid bfloat16 values
        // We iterate through all 65536 possible bit patterns
        std::vector<Bfloat16Entry> valid_entries;
        valid_entries.reserve(TOTAL_PATTERNS);  // Avoid reallocations

        for (uint32_t bits = 0; bits < TOTAL_PATTERNS; ++bits) {
            uint16_t bits16 = static_cast<uint16_t>(bits);
            std::bfloat16_t value = bits_to_bfloat16(bits16);

            // Only include finite values (exclude NaN and Inf)
            if (is_finite_bfloat16(value)) {
                valid_entries.push_back({value, bits16});
            }
        }

        valid_count = valid_entries.size();

        // Step 3: Sort valid entries by numerical value
        // This uses the comparison operator defined in Bfloat16Entry
        // which compares float representations for correct ordering
        std::sort(valid_entries.begin(), valid_entries.end());

        // Step 4 & 5: Assign ULP indices and populate lookup table
        // Equal values get the same index - this handles the +0/-0 case
        // and any other values that might compare equal
        int64_t current_index = 0;

        for (size_t i = 0; i < valid_entries.size(); ++i) {
            if (i > 0) {
                // Check if current value is different from previous
                // We compare as floats for proper numerical comparison
                float prev = static_cast<float>(valid_entries[i - 1].value);
                float curr = static_cast<float>(valid_entries[i].value);

                // Only increment index if values are different
                // This ensures +0 and -0 (which are equal) share the same index
                if (curr != prev) {
                    ++current_index;
                }
            }

            // Store the ULP index for this bit pattern
            ulp_index_table[valid_entries[i].bits] = current_index;
        }
    }
};

/**
 * @class SimpleUlpCalculator
 * @brief Simple ULP calculator for verification purposes
 *
 * This class provides a straightforward, easy-to-verify ULP calculation
 * without the complexity of the lookup table approach. It's used to
 * verify that UlpCalculator produces correct results.
 *
 * ## Algorithm
 *
 * Instead of precomputing all ULP indices, this calculator counts the
 * number of representable values between two given values on-demand.
 * This is slower but simpler and easier to verify for correctness.
 *
 * ## Limitations
 *
 * - Slower for repeated queries (O(n) per query vs O(1) for UlpCalculator)
 * - Still handles +0/-0 correctly
 * - Excludes NaN and Inf from counting
 */
class SimpleUlpCalculator {
public:
    /**
     * @brief Calculate ULP distance by counting values between a and b
     *
     * This method iterates through all possible bfloat16 values and counts
     * how many fall strictly between min(a,b) and max(a,b), then adds 1.
     *
     * The "+1" is because ULP distance includes one endpoint:
     *   - distance(x, x) = 0
     *   - distance(x, next_after(x)) = 1
     *
     * @param a First bfloat16 value
     * @param b Second bfloat16 value
     * @return ULP distance, or -1 if either value is non-numeric
     */
    static int64_t ulp_distance(std::bfloat16_t a, std::bfloat16_t b) {
        // Check for invalid inputs
        if (!is_finite_bfloat16(a) || !is_finite_bfloat16(b)) {
            return -1;
        }

        // Convert to float for comparison
        float fa = static_cast<float>(a);
        float fb = static_cast<float>(b);

        // Handle equal values (including +0 == -0)
        if (fa == fb) {
            return 0;
        }

        // Ensure fa < fb for consistent counting
        if (fa > fb) {
            std::swap(fa, fb);
        }

        // Count distinct values in the range (fa, fb]
        // We use a set-like approach to handle equal values correctly
        int64_t count = 0;
        float prev_value = fa;
        bool started = false;

        // We need to iterate in sorted order, so we collect and sort first
        std::vector<float> sorted_values;
        sorted_values.reserve(65536);

        for (uint32_t bits = 0; bits < 65536; ++bits) {
            std::bfloat16_t val = bits_to_bfloat16(static_cast<uint16_t>(bits));
            if (is_finite_bfloat16(val)) {
                float fval = static_cast<float>(val);
                // Only include values in the range (fa, fb]
                if (fval > fa && fval <= fb) {
                    sorted_values.push_back(fval);
                }
            }
        }

        // Sort the values
        std::sort(sorted_values.begin(), sorted_values.end());

        // Count distinct values
        for (size_t i = 0; i < sorted_values.size(); ++i) {
            if (i == 0 || sorted_values[i] != sorted_values[i - 1]) {
                ++count;
            }
        }

        return count;
    }
};

/**
 * @brief Print information about a bfloat16 value
 *
 * Utility function for debugging that shows the value, its bit pattern
 * in hexadecimal, and its ULP index.
 *
 * @param value The bfloat16 value to inspect
 * @param calc The ULP calculator for looking up the index
 */
void print_bfloat16_info(std::bfloat16_t value, const UlpCalculator& calc) {
    uint16_t bits = bfloat16_to_bits(value);
    float fval = static_cast<float>(value);
    int64_t ulp_idx = calc.get_ulp_index(value);

    std::cout << "Value: " << std::setw(15) << std::setprecision(10) << fval
              << " | Bits: 0x" << std::hex << std::setw(4) << std::setfill('0') << bits
              << std::dec << std::setfill(' ')
              << " | ULP Index: " << std::setw(6) << ulp_idx
              << std::endl;
}

/**
 * @brief Verify the ULP calculator against the simple implementation
 *
 * This function tests a sample of value pairs to ensure that both
 * ULP calculation methods produce the same results.
 *
 * @param calc The ULP calculator to verify
 * @return true if all tests pass, false otherwise
 */
bool verify_ulp_calculator(const UlpCalculator& calc) {
    std::cout << "\n=== Verifying ULP Calculator ===\n" << std::endl;

    bool all_passed = true;
    int tests_run = 0;
    int tests_passed = 0;

    // Test cases: pairs of bit patterns to compare
    // We test various interesting cases:
    std::vector<std::pair<uint16_t, uint16_t>> test_cases = {
        // Case 1: Same value
        {0x3F80, 0x3F80},  // 1.0 and 1.0

        // Case 2: Positive and negative zero
        {0x0000, 0x8000},  // +0 and -0

        // Case 3: Adjacent positive values
        {0x3F80, 0x3F81},  // 1.0 and next value after 1.0

        // Case 4: Adjacent negative values
        {0xBF80, 0xBF81},  // -1.0 and next value after -1.0 (toward -inf)

        // Case 5: Across zero
        {0x8001, 0x0001},  // Smallest negative and smallest positive subnormal

        // Case 6: Larger gap
        {0x3F80, 0x4000},  // 1.0 and 2.0

        // Case 7: Negative values
        {0xC000, 0xBF80},  // -2.0 and -1.0
    };

    for (const auto& [bits_a, bits_b] : test_cases) {
        std::bfloat16_t a = bits_to_bfloat16(bits_a);
        std::bfloat16_t b = bits_to_bfloat16(bits_b);

        int64_t dist_fast = calc.ulp_distance(a, b);
        int64_t dist_simple = SimpleUlpCalculator::ulp_distance(a, b);

        ++tests_run;
        bool passed = (dist_fast == dist_simple);
        if (passed) ++tests_passed;

        std::cout << "Test: 0x" << std::hex << std::setw(4) << std::setfill('0') << bits_a
                  << " vs 0x" << std::setw(4) << bits_b << std::dec << std::setfill(' ')
                  << " | Fast: " << std::setw(6) << dist_fast
                  << " | Simple: " << std::setw(6) << dist_simple
                  << " | " << (passed ? "PASS" : "FAIL")
                  << std::endl;

        if (!passed) all_passed = false;
    }

    std::cout << "\nResults: " << tests_passed << "/" << tests_run << " tests passed\n";

    return all_passed;
}

/**
 * @brief Display statistics about the bfloat16 number system
 *
 * @param calc The ULP calculator to query
 */
void print_bfloat16_statistics(const UlpCalculator& calc) {
    std::cout << "\n=== BFloat16 Statistics ===\n" << std::endl;

    std::cout << "Total 16-bit patterns:     " << UlpCalculator::TOTAL_PATTERNS << std::endl;
    std::cout << "Valid (finite) values:     " << calc.get_valid_count() << std::endl;
    std::cout << "Distinct ULP indices:      " << (calc.get_max_ulp_index() + 1) << std::endl;
    std::cout << "Invalid patterns (NaN/Inf): "
              << (UlpCalculator::TOTAL_PATTERNS - calc.get_valid_count()) << std::endl;

    // Show some special values
    std::cout << "\n--- Special Values ---\n" << std::endl;

    std::cout << "Positive zero: ";
    print_bfloat16_info(bits_to_bfloat16(0x0000), calc);

    std::cout << "Negative zero: ";
    print_bfloat16_info(bits_to_bfloat16(0x8000), calc);

    std::cout << "One:           ";
    print_bfloat16_info(1.0bf16, calc);

    std::cout << "Negative one:  ";
    print_bfloat16_info(-1.0bf16, calc);

    std::cout << "Smallest pos subnormal: ";
    print_bfloat16_info(bits_to_bfloat16(0x0001), calc);

    std::cout << "Largest finite:         ";
    print_bfloat16_info(bits_to_bfloat16(0x7F7F), calc);
}

/**
 * @brief Main entry point - demonstrates the ULP calculator
 */
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   BFloat16 ULP Calculator Reference   " << std::endl;
    std::cout << "========================================" << std::endl;

    // Build the ULP calculator
    std::cout << "\nBuilding ULP lookup table..." << std::endl;
    UlpCalculator calc;
    std::cout << "Done!" << std::endl;

    // Print statistics
    print_bfloat16_statistics(calc);

    // Verify the implementation
    bool verified = verify_ulp_calculator(calc);

    if (verified) {
        std::cout << "\n[SUCCESS] ULP Calculator verified successfully!\n" << std::endl;
    } else {
        std::cout << "\n[ERROR] ULP Calculator verification failed!\n" << std::endl;
        return 1;
    }

    // Demonstrate usage with some example calculations
    std::cout << "\n=== Example ULP Calculations ===\n" << std::endl;

    std::bfloat16_t one = 1.0bf16;
    std::bfloat16_t two = 2.0bf16;
    std::bfloat16_t half = 0.5bf16;

    std::cout << "ULP distance between 1.0 and 2.0: "
              << calc.ulp_distance(one, two) << std::endl;
    std::cout << "ULP distance between 1.0 and 0.5: "
              << calc.ulp_distance(one, half) << std::endl;
    std::cout << "ULP distance between +0 and -0:   "
              << calc.ulp_distance(bits_to_bfloat16(0x0000), bits_to_bfloat16(0x8000))
              << std::endl;

    return 0;
}
