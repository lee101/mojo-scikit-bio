"""Sequence distance and ecological diversity kernels."""

from std.algorithm.functional import parallelize
from std.math import exp, log, pow, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BW = simd_width_of[DType.uint8]()
comptime FW = simd_width_of[DType.float64]()


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


@export("msb_hamming")
def msb_hamming(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Int:
    var a = bp(a_addr)
    var b = bp(b_addr)
    var distance = 0
    var i = 0
    var ones = SIMD[DType.uint8, BW](1)
    var zeros = SIMD[DType.uint8, BW](0)
    while i + BW <= n:
        distance += Int(
            a.load[width=BW](i).ne(b.load[width=BW](i)).select(
                ones, zeros
            ).reduce_add()[0]
        )
        i += BW
    while i < n:
        if a[i] != b[i]:
            distance += 1
        i += 1
    return distance


def kmer_hash(seq: BPtr, start: Int, k: Int) -> UInt64:
    var value = UInt64(14695981039346656037)
    for i in range(k):
        value = (value ^ UInt64(seq[start + i])) * UInt64(1099511628211)
    return value


def same_kmer(
    a: BPtr,
    b: BPtr,
    representative_source: Int,
    representative_start: Int,
    current_source: Int,
    current_start: Int,
    k: Int,
) -> Bool:
    for i in range(k):
        var left = (
            a[representative_start + i]
            if representative_source == 1
            else b[representative_start + i]
        )
        var right = (
            a[current_start + i]
            if current_source == 1
            else b[current_start + i]
        )
        if left != right:
            return False
    return True


def locate_kmer(
    a: BPtr,
    b: BPtr,
    hashes: U64Ptr,
    starts: I64Ptr,
    sources: BPtr,
    owners: BPtr,
    capacity: Int,
    current_source: Int,
    current_start: Int,
    k: Int,
    hash_value: UInt64,
) -> Int:
    var slot = Int(hash_value & UInt64(capacity - 1))
    while owners[slot] != 0:
        if (
            hashes[slot] == hash_value
            and same_kmer(
                a,
                b,
                Int(sources[slot]),
                Int(starts[slot]),
                current_source,
                current_start,
                k,
            )
        ):
            return slot
        slot = (slot + 1) & (capacity - 1)
    return slot


@export("msb_kmer_distance")
def msb_kmer_distance(
    a_addr: Int,
    na: Int,
    b_addr: Int,
    nb: Int,
    k: Int,
    step: Int,
    hashes_addr: Int,
    starts_addr: Int,
    sources_addr: Int,
    owners_addr: Int,
    capacity: Int,
) abi("C") -> Float64:
    var a = bp(a_addr)
    var b = bp(b_addr)
    var hashes = u64p(hashes_addr)
    var starts = i64p(starts_addr)
    var sources = bp(sources_addr)
    var owners = bp(owners_addr)
    var union_count = 0
    var shared_count = 0

    var start = 0
    while start + k <= na:
        var hash_value = kmer_hash(a, start, k)
        var slot = locate_kmer(
            a, b, hashes, starts, sources, owners, capacity, 1, start, k, hash_value
        )
        if owners[slot] == 0:
            hashes[slot] = hash_value
            starts[slot] = Int64(start)
            sources[slot] = 1
            owners[slot] = 1
            union_count += 1
        start += step

    start = 0
    while start + k <= nb:
        var hash_value = kmer_hash(b, start, k)
        var slot = locate_kmer(
            a, b, hashes, starts, sources, owners, capacity, 2, start, k, hash_value
        )
        if owners[slot] == 0:
            hashes[slot] = hash_value
            starts[slot] = Int64(start)
            sources[slot] = 2
            owners[slot] = 2
            union_count += 1
        elif owners[slot] == 1:
            owners[slot] = 3
            shared_count += 1
        start += step

    if union_count == 0:
        return nan_value()
    return Float64(union_count - shared_count) / Float64(union_count)


def alpha_value(
    counts: FPtr,
    n: Int,
    code: Int,
    parameter1: Float64,
    parameter2: Float64,
    flag: Int,
) -> Float64:
    var total = 0.0
    var sum_squares = 0.0
    var sum_count_log_count = 0.0
    var maximum = 0.0
    var observed = 0
    var singleton_count = 0
    var doubleton_count = 0
    for i in range(n):
        var value = counts[i]
        if value > 0.0:
            total += value
            sum_squares += value * value
            sum_count_log_count += value * log(value)
            observed += 1
            if value > maximum:
                maximum = value
            if value == 1.0:
                singleton_count += 1
            elif value == 2.0:
                doubleton_count += 1

    if code == 0:
        return Float64(observed)
    if code == 1:
        return Float64(singleton_count)
    if code == 2:
        return Float64(doubleton_count)
    if code == 3:
        if flag == 0 and singleton_count > 0 and doubleton_count > 0:
            return Float64(observed) + Float64(singleton_count * singleton_count) / (
                2.0 * Float64(doubleton_count)
            )
        return Float64(observed) + Float64(singleton_count * (singleton_count - 1)) / (
            2.0 * Float64(doubleton_count + 1)
        )
    if observed == 0:
        return nan_value()
    if code == 4:
        return maximum / total

    var dominance_value = sum_squares / (total * total)
    if flag != 0:
        dominance_value = (sum_squares - total) / (total * (total - 1.0))
    if code == 5:
        return dominance_value
    if code == 7:
        return 1.0 - dominance_value
    if code == 8:
        return 1.0 / dominance_value
    if code == 9:
        return 1.0 - Float64(singleton_count) / total
    if code == 10:
        if total == 1.0:
            return nan_value()
        return Float64(observed - 1) / log(total)
    if code == 11:
        return Float64(observed) / sqrt(total)
    if code == 12:
        if total == 1.0:
            return nan_value()
        return (total - sqrt(sum_squares)) / (total - sqrt(total))
    if code == 13:
        return sqrt(sum_squares) / sqrt(
            (total - Float64(observed) + 1.0)
            * (total - Float64(observed) + 1.0)
            + Float64(observed - 1)
        )
    if code == 15:
        return 1.0 / (Float64(observed) * dominance_value)
    if code == 17:
        return Float64(singleton_count) / total

    var entropy = log(total) - sum_count_log_count / total
    if code == 6:
        if flag != 0:
            return exp(entropy)
        return entropy if parameter1 == 0.0 else entropy / log(parameter1)
    if code == 14:
        if observed == 1:
            return 1.0
        var base_log = log(Float64(observed))
        if parameter1 != 0.0:
            base_log /= log(parameter1)
            entropy /= log(parameter1)
        return entropy / base_log
    if code == 16:
        if observed == 1:
            return 1.0
        return (exp(entropy) - 1.0) / Float64(observed - 1)

    if observed == 1:
        return 0.0 if code == 19 or code == 20 else 1.0
    if flag != 0:
        if code == 18:
            return total / maximum
        if code == 19:
            var answer = -log(maximum / total)
            return answer if parameter2 == 0.0 else answer / log(parameter2)
        return 0.0
    if parameter1 == 1.0:
        if code == 18:
            return exp(entropy)
        if code == 19:
            return entropy if parameter2 == 0.0 else entropy / log(parameter2)
        return entropy

    var power_sum = 0.0
    for i in range(n):
        var value = counts[i]
        if value > 0.0:
            power_sum += pow(value / total, parameter1)
    if code == 18:
        return pow(power_sum, 1.0 / (1.0 - parameter1))
    if code == 19:
        var answer = log(power_sum) / (1.0 - parameter1)
        return answer if parameter2 == 0.0 else answer / log(parameter2)
    return (1.0 - power_sum) / (parameter1 - 1.0)


@export("msb_alpha")
def msb_alpha(
    counts_addr: Int,
    n: Int,
    code: Int,
    parameter1: Float64,
    parameter2: Float64,
    flag: Int,
) abi("C") -> Float64:
    return alpha_value(fp(counts_addr), n, code, parameter1, parameter2, flag)


@export("msb_alpha_batch")
def msb_alpha_batch(
    counts_addr: Int,
    rows: Int,
    columns: Int,
    result_addr: Int,
    code: Int,
    parameter1: Float64,
    parameter2: Float64,
    flag: Int,
) abi("C"):
    var counts = fp(counts_addr)
    var result = fp(result_addr)

    def calculate(row: Int) {imm}:
        result[row] = alpha_value(
            counts + row * columns, columns, code, parameter1, parameter2, flag
        )

    if rows * columns >= 100_000:
        parallelize(calculate, rows, min(rows, 16))
    else:
        for row in range(rows):
            calculate(row)


def braycurtis_pair(a: FPtr, b: FPtr, n: Int) -> Float64:
    var numerator = SIMD[DType.float64, FW](0.0)
    var denominator = SIMD[DType.float64, FW](0.0)
    var i = 0
    while i + FW <= n:
        var left = a.load[width=FW](i)
        var right = b.load[width=FW](i)
        numerator += abs(left - right)
        denominator += abs(left + right)
        i += FW

    var numerator_sum = numerator.reduce_add()[0]
    var denominator_sum = denominator.reduce_add()[0]
    while i < n:
        numerator_sum += abs(a[i] - b[i])
        denominator_sum += abs(a[i] + b[i])
        i += 1
    return numerator_sum / denominator_sum


def beta_pair(a: FPtr, b: FPtr, n: Int, code: Int, parameter: Float64) -> Float64:
    if code == 3:
        return braycurtis_pair(a, b, n)

    var sum_a = 0.0
    var sum_b = 0.0
    var sum_a2 = 0.0
    var sum_b2 = 0.0
    var dot = 0.0
    var accumulated = 0.0
    var maximum = 0.0
    var both = 0.0
    var neither = 0.0
    var a_only = 0.0
    var b_only = 0.0
    for i in range(n):
        var left = a[i]
        var right = b[i]
        var difference = abs(left - right)
        sum_a += left
        sum_b += right
        sum_a2 += left * left
        sum_b2 += right * right
        dot += left * right
        if code == 0 or code == 1:
            accumulated += difference * difference
        elif code == 2:
            accumulated += difference
        elif code == 4:
            var denominator = abs(left) + abs(right)
            if denominator != 0.0:
                accumulated += difference / denominator
        elif code == 5:
            if difference > maximum:
                maximum = difference
        elif code == 8:
            accumulated += 1.0 if left != right else 0.0
        elif code == 18:
            accumulated += pow(difference, parameter)
        var left_present = left != 0.0
        var right_present = right != 0.0
        if left_present and right_present:
            both += 1.0
        elif left_present:
            a_only += 1.0
        elif right_present:
            b_only += 1.0
        else:
            neither += 1.0

    if code == 0:
        return sqrt(accumulated)
    if code == 1:
        return accumulated
    if code == 2:
        return accumulated
    if code == 4:
        return accumulated
    if code == 5:
        return maximum
    if code == 6:
        var centered = dot - sum_a * sum_b / Float64(n)
        var norm_a = sum_a2 - sum_a * sum_a / Float64(n)
        var norm_b = sum_b2 - sum_b * sum_b / Float64(n)
        return 1.0 - centered / sqrt(norm_a * norm_b)
    if code == 7:
        return 1.0 - dot / sqrt(sum_a2 * sum_b2)
    if code == 8:
        return accumulated / Float64(n)
    if code == 11:
        return (a_only + b_only) / Float64(n)
    if code == 9:
        return (a_only + b_only) / (both + a_only + b_only)
    if code == 10:
        return (a_only + b_only) / (2.0 * both + a_only + b_only)
    if code == 12:
        var differences = a_only + b_only
        return 2.0 * differences / (both + neither + 2.0 * differences)
    if code == 13:
        return (Float64(n) - both) / Float64(n)
    if code == 14:
        return 2.0 * (a_only + b_only) / (both + 2.0 * (a_only + b_only))
    if code == 15:
        var cross = a_only * b_only
        var denominator = both * neither + cross
        return 0.0 if denominator == 0.0 else 2.0 * cross / denominator
    if code == 18:
        return pow(accumulated, 1.0 / parameter)
    if code == 19:
        accumulated = 0.0
        for i in range(n):
            var left = a[i] / sum_a
            var right = b[i] / sum_b
            var middle = 0.5 * (left + right)
            if left > 0.0:
                accumulated += 0.5 * left * log(left / middle)
            if right > 0.0:
                accumulated += 0.5 * right * log(right / middle)
        return sqrt(accumulated)
    return nan_value()


@export("msb_beta")
def msb_beta(
    counts_addr: Int,
    rows: Int,
    columns: Int,
    result_addr: Int,
    code: Int,
    parameter: Float64,
) abi("C"):
    var counts = fp(counts_addr)
    var result = fp(result_addr)

    def calculate_row(row: Int) {imm}:
        result[row * rows + row] = 0.0
        for other in range(row + 1, rows):
            var value = beta_pair(
                counts + row * columns,
                counts + other * columns,
                columns,
                code,
                parameter,
            )
            result[row * rows + other] = value
            result[other * rows + row] = value

    if rows * rows * columns >= 250_000:
        parallelize(calculate_row, rows, min(rows, 16))
    else:
        for row in range(rows):
            calculate_row(row)
