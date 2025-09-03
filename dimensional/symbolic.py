#!/usr/bin/env python3
"""Symbolic mathematical compression for dimensional data."""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"_type": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, SymbolicPattern):
            return {
                "_type": "pattern",
                "pattern_type": obj.pattern_type,
                "parameters": obj.parameters,
                "compression_ratio": obj.compression_ratio,
                "reconstruction_error": obj.reconstruction_error
            }
        return super().default(obj)


class NumpyDecoder(json.JSONDecoder):
    """JSON decoder for NumPy arrays."""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" in obj:
            if obj["_type"] == "ndarray":
                return np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
            if obj["_type"] == "pattern":
                return SymbolicPattern(
                    pattern_type=obj["pattern_type"],
                    parameters=obj["parameters"],
                    compression_ratio=obj["compression_ratio"],
                    reconstruction_error=obj["reconstruction_error"]
                )
        return obj


@dataclass
class SymbolicPattern:
    """Detected symbolic mathematical pattern."""
    pattern_type: str
    parameters: dict[str, Any]
    compression_ratio: float
    reconstruction_error: float

class SymbolicMathematicalCompressor:
    """Symbolic compression for mathematical data using pattern recognition."""

    def __init__(self, precision_threshold: float = 1e-14) -> None:
        self.precision_threshold = precision_threshold

        self.constants = {
            "pi": np.pi, "e": np.e, "phi": (1 + np.sqrt(5)) / 2, "psi": (np.sqrt(5) - 1) / 2,
            "sqrt2": np.sqrt(2), "sqrt3": np.sqrt(3), "sqrt5": np.sqrt(5),
            "ln2": np.log(2), "ln10": np.log(10), "euler_gamma": 0.5772156649015329,
            "catalan": 0.9159655941772190,
        }
        self.dimensional_constants = {
            "pi_half": np.pi / 2, "pi_squared": np.pi**2, "two_pi": 2 * np.pi,
            "pi_cubed": np.pi**3, "sqrt_pi": np.sqrt(np.pi), "sqrt_2pi": np.sqrt(2 * np.pi),
            "gamma_half": 1.7724538509055159, "log_pi": np.log(np.pi), "log_2pi": np.log(2 * np.pi),
        }
        self.all_constants = {**self.constants, **self.dimensional_constants}
        self.pattern_min_length = 5
        self.similarity_threshold = 1e-12

    def compress_array_symbolic(
        self,
        data: NDArray,
        detect_patterns: bool = True,
        max_polynomial_degree: int = 5,
    ) -> tuple[bytes, dict[str, Any]]:
        """Symbolically compress a mathematical array using pattern detection."""
        start_time = time.perf_counter()

        patterns = []
        compressed_representation = {"original_shape": data.shape}
        remaining_data = data.copy().flatten()

        if detect_patterns:
            constant_pattern = self._detect_constants(remaining_data)
            if constant_pattern and constant_pattern.compression_ratio > 1.1:
                patterns.append(constant_pattern)
                remaining_data = self._remove_constant_pattern(remaining_data, constant_pattern)
            if len(remaining_data) >= self.pattern_min_length:
                arithmetic_pattern = self._detect_arithmetic_progression(remaining_data)
                if arithmetic_pattern and arithmetic_pattern.compression_ratio > 1.2:
                    patterns.append(arithmetic_pattern)
                    remaining_data = self._remove_arithmetic_pattern(remaining_data, arithmetic_pattern)
            if len(remaining_data) >= self.pattern_min_length:
                geometric_pattern = self._detect_geometric_progression(remaining_data)
                if geometric_pattern and geometric_pattern.compression_ratio > 1.2:
                    patterns.append(geometric_pattern)
                    remaining_data = self._remove_geometric_pattern(remaining_data, geometric_pattern)
            if len(remaining_data) >= self.pattern_min_length:
                poly_pattern = self._detect_polynomial_pattern(remaining_data, max_polynomial_degree)
                if poly_pattern and poly_pattern.compression_ratio > 1.3:
                    patterns.append(poly_pattern)
                    remaining_data = self._remove_polynomial_pattern(remaining_data, poly_pattern)
            if len(remaining_data) >= self.pattern_min_length:
                repeat_pattern = self._detect_repetitive_pattern(remaining_data)
                if repeat_pattern and repeat_pattern.compression_ratio > 1.5:
                    patterns.append(repeat_pattern)
                    remaining_data = self._remove_repetitive_pattern(remaining_data, repeat_pattern)

        compressed_representation["patterns"] = patterns
        compressed_representation["remaining_data"] = remaining_data
        compressed_representation["compression_metadata"] = {
            "num_patterns": len(patterns), "original_size": data.size,
            "remaining_size": len(remaining_data), "pattern_types": [p.pattern_type for p in patterns],
        }

        # Use JSON for safe serialization instead of pickle
        json_data = json.dumps(compressed_representation, cls=NumpyEncoder)
        compressed_data = json_data.encode('utf-8')

        compression_time = time.perf_counter() - start_time
        original_size = data.nbytes
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        try:
            reconstructed, _ = self.decompress_array_symbolic(compressed_data)
            reconstruction_error = self._calculate_reconstruction_error(data, reconstructed)
        except Exception:
            reconstruction_error = float("inf")

        metadata = {
            "compression_method": "symbolic_mathematical", "original_size": original_size,
            "compressed_size": compressed_size, "compression_ratio": compression_ratio,
            "compression_time": compression_time, "reconstruction_error": reconstruction_error,
            "patterns_detected": len(patterns), "pattern_details": [
                {"type": p.pattern_type, "compression_ratio": p.compression_ratio, "error": p.reconstruction_error}
                for p in patterns
            ],
        }

        return compressed_data, metadata

    def decompress_array_symbolic(
        self, compressed_data: bytes
    ) -> tuple[NDArray, dict[str, Any]]:
        """Decompress symbolically compressed array."""
        start_time = time.perf_counter()

        # Safe JSON deserialization
        json_data = compressed_data.decode('utf-8')
        compressed_representation = json.loads(json_data, cls=NumpyDecoder)

        patterns = compressed_representation["patterns"]
        remaining_data = compressed_representation["remaining_data"]
        original_shape = compressed_representation["original_shape"]

        reconstructed = remaining_data.copy()

        for pattern in reversed(patterns):
            reconstructed = self._apply_pattern(reconstructed, pattern)
        reconstructed = reconstructed.reshape(original_shape)

        decompression_time = time.perf_counter() - start_time
        metadata = {
            "decompression_time": decompression_time, "patterns_applied": len(patterns),
            "reconstruction_method": "symbolic_pattern_application",
        }

        return reconstructed, metadata

    def _detect_constants(self, data: NDArray) -> Optional[SymbolicPattern]:
        """Detect mathematical constants in the data."""
        if len(data) < 2:
            return None

        constant_matches = {}
        tolerance = self.similarity_threshold

        for const_name, const_value in self.all_constants.items():
            matches = np.isclose(
                data, const_value, rtol=tolerance, atol=tolerance
            )
            match_count = np.sum(matches)

            if match_count >= 2:
                constant_matches[const_name] = {
                    "value": const_value, "matches": matches, "count": match_count,
                }

        if not constant_matches:
            return None

        best_const = max(constant_matches.items(), key=lambda x: x[1]["count"])
        const_name, const_info = best_const

        original_bits = const_info["count"] * 64
        compressed_bits = len(const_name) * 8 + const_info["count"] * 1
        compression_ratio = original_bits / max(compressed_bits, 1)

        matched_values = data[const_info["matches"]]
        reconstruction_error = np.max(
            np.abs(matched_values - const_info["value"])
        )

        return SymbolicPattern(
            pattern_type="mathematical_constant",
            parameters={
                "constant_name": const_name, "constant_value": const_info["value"],
                "indices": np.where(const_info["matches"])[0], "count": const_info["count"],
            },
            compression_ratio=compression_ratio, reconstruction_error=reconstruction_error,
        )

    def _detect_arithmetic_progression(
        self, data: NDArray
    ) -> Optional[SymbolicPattern]:
        """Detect arithmetic progressions in the data."""
        if len(data) < self.pattern_min_length:
            return None

        best_pattern = None
        best_ratio = 1.0

        for start_idx in range(len(data) - self.pattern_min_length + 1):
            for length in range(
                self.pattern_min_length, min(len(data) - start_idx + 1, 100)
            ):
                segment = data[start_idx : start_idx + length]

                if len(segment) < 3:
                    continue

                diffs = np.diff(segment)
                if np.allclose(diffs, diffs[0], rtol=1e-10, atol=1e-15):
                    first_term = segment[0]
                    common_diff = diffs[0]

                    original_bits = length * 64
                    compressed_bits = 3 * 64 + len(str(start_idx)) * 8
                    compression_ratio = original_bits / max(compressed_bits, 1)

                    if compression_ratio > best_ratio:
                        reconstructed = first_term + common_diff * np.arange(
                            length
                        )
                        error = np.max(np.abs(segment - reconstructed))

                        best_pattern = SymbolicPattern(
                            pattern_type="arithmetic_progression",
                            parameters={
                                "start_index": start_idx,
                                "first_term": first_term,
                                "common_difference": common_diff,
                                "length": length,
                            },
                            compression_ratio=compression_ratio,
                            reconstruction_error=error,
                        )
                        best_ratio = compression_ratio

        return best_pattern

    def _detect_geometric_progression(
        self, data: NDArray
    ) -> Optional[SymbolicPattern]:
        """Detect geometric progressions in the data."""
        if len(data) < self.pattern_min_length:
            return None

        positive_mask = data > 0
        if np.sum(positive_mask) < self.pattern_min_length:
            return None

        positive_data = data[positive_mask]
        positive_indices = np.where(positive_mask)[0]

        best_pattern = None
        best_ratio = 1.0

        for start_idx in range(
            len(positive_data) - self.pattern_min_length + 1
        ):
            for length in range(
                self.pattern_min_length,
                min(len(positive_data) - start_idx + 1, 50),
            ):
                segment = positive_data[start_idx : start_idx + length]

                if len(segment) < 3:
                    continue

                ratios = segment[1:] / segment[:-1]
                if np.allclose(ratios, ratios[0], rtol=1e-10, atol=1e-15):
                    first_term = segment[0]
                    common_ratio = ratios[0]

                    original_bits = length * 64
                    compressed_bits = 4 * 64
                    compression_ratio = original_bits / max(compressed_bits, 1)

                    if compression_ratio > best_ratio:
                        reconstructed = first_term * (
                            common_ratio ** np.arange(length)
                        )
                        error = np.max(np.abs(segment - reconstructed))

                        best_pattern = SymbolicPattern(
                            pattern_type="geometric_progression",
                            parameters={
                                "original_start_index": start_idx,
                                "indices_in_original": positive_indices[
                                    start_idx : start_idx + length
                                ],
                                "first_term": first_term,
                                "common_ratio": common_ratio,
                                "length": length,
                            },
                            compression_ratio=compression_ratio,
                            reconstruction_error=error,
                        )
                        best_ratio = compression_ratio

        return best_pattern

    def _detect_polynomial_pattern(
        self, data: NDArray, max_degree: int
    ) -> Optional[SymbolicPattern]:
        """Detect polynomial patterns in the data."""
        if len(data) < max_degree + 2:
            return None

        best_pattern = None
        best_ratio = 1.0

        x = np.arange(len(data))

        for degree in range(1, min(max_degree + 1, len(data) - 1)):
            try:
                coeffs = np.polyfit(x, data, degree)
                poly_fit = np.polyval(coeffs, x)

                mse = np.mean((data - poly_fit) ** 2)
                max_error = np.max(np.abs(data - poly_fit))

                if max_error < self.precision_threshold:
                    original_bits = len(data) * 64
                    compressed_bits = len(coeffs) * 64 + 32
                    compression_ratio = original_bits / max(compressed_bits, 1)

                    if compression_ratio > best_ratio:
                        best_pattern = SymbolicPattern(
                            pattern_type="polynomial",
                            parameters={"coefficients": coeffs, "degree": degree, "length": len(data), "mse": mse},
                            compression_ratio=compression_ratio, reconstruction_error=max_error,
                        )
                        best_ratio = compression_ratio

            except np.linalg.LinAlgError:
                continue

        return best_pattern

    def _detect_repetitive_pattern(
        self, data: NDArray
    ) -> Optional[SymbolicPattern]:
        """Detect repetitive patterns in the data."""
        if len(data) < 2 * self.pattern_min_length:
            return None

        best_pattern = None
        best_ratio = 1.0

        for pattern_length in range(2, len(data) // 2 + 1):
            if len(data) % pattern_length != 0:
                continue

            pattern_unit = data[:pattern_length]
            num_repeats = len(data) // pattern_length

            reconstructed = np.tile(pattern_unit, num_repeats)

            if np.allclose(data, reconstructed, rtol=1e-12, atol=1e-15):
                original_bits = len(data) * 64
                compressed_bits = pattern_length * 64 + 32
                compression_ratio = original_bits / max(compressed_bits, 1)

                if compression_ratio > best_ratio:
                    error = np.max(np.abs(data - reconstructed))

                    best_pattern = SymbolicPattern(
                        pattern_type="repetitive",
                        parameters={"pattern_unit": pattern_unit, "pattern_length": pattern_length, "num_repeats": num_repeats},
                        compression_ratio=compression_ratio, reconstruction_error=error,
                    )
                    best_ratio = compression_ratio

        return best_pattern

    def _remove_constant_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Remove constant pattern from data."""
        indices = pattern.parameters["indices"]
        mask = np.ones(len(data), dtype=bool)
        mask[indices] = False
        return data[mask]

    def _remove_arithmetic_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Remove arithmetic progression pattern from data."""
        start_idx = pattern.parameters["start_index"]
        length = pattern.parameters["length"]
        mask = np.ones(len(data), dtype=bool)
        mask[start_idx : start_idx + length] = False
        return data[mask]

    def _remove_geometric_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Remove geometric progression pattern from data."""
        indices = pattern.parameters["indices_in_original"]
        mask = np.ones(len(data), dtype=bool)
        mask[indices] = False
        return data[mask]

    def _remove_polynomial_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Remove polynomial pattern from data."""
        return np.array([])

    def _remove_repetitive_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Remove repetitive pattern from data."""
        return np.array([])

    def _apply_pattern(
        self, data: NDArray, pattern: SymbolicPattern
    ) -> NDArray:
        """Apply a pattern to reconstruct original data."""
        if pattern.pattern_type == "mathematical_constant":
            const_value = pattern.parameters["constant_value"]
            indices = pattern.parameters["indices"]

            result = np.zeros(len(data) + len(indices))

            data_mask = np.ones(len(result), dtype=bool)
            data_mask[indices] = False
            result[data_mask] = data

            result[indices] = const_value

            return result

        elif pattern.pattern_type == "arithmetic_progression":
            start_idx = pattern.parameters["start_index"]
            first_term = pattern.parameters["first_term"]
            common_diff = pattern.parameters["common_difference"]
            length = pattern.parameters["length"]

            progression = first_term + common_diff * np.arange(length)

            result = np.zeros(len(data) + length)
            result[:start_idx] = (
                data[:start_idx] if start_idx < len(data) else []
            )
            result[start_idx : start_idx + length] = progression
            result[start_idx + length :] = (
                data[start_idx:] if start_idx < len(data) else []
            )

            return result

        elif pattern.pattern_type == "geometric_progression":
            indices = pattern.parameters["indices_in_original"]
            first_term = pattern.parameters["first_term"]
            common_ratio = pattern.parameters["common_ratio"]
            length = pattern.parameters["length"]

            progression = first_term * (common_ratio ** np.arange(length))

            result = np.zeros(len(data) + length)
            data_mask = np.ones(len(result), dtype=bool)
            data_mask[indices] = False
            result[data_mask] = data
            result[indices] = progression

            return result

        elif pattern.pattern_type == "polynomial":
            coeffs = pattern.parameters["coefficients"]
            length = pattern.parameters["length"]
            x = np.arange(length)
            return np.polyval(coeffs, x)

        elif pattern.pattern_type == "repetitive":
            pattern_unit = pattern.parameters["pattern_unit"]
            num_repeats = pattern.parameters["num_repeats"]
            return np.tile(pattern_unit, num_repeats)

        return data

    def _calculate_reconstruction_error(
        self, original: NDArray, reconstructed: NDArray
    ) -> float:
        """Calculate reconstruction error between original and reconstructed data."""
        if original.shape != reconstructed.shape:
            return float("inf")

        diff = np.abs(original - reconstructed)
        relative_error = diff / np.maximum(np.abs(original), 1e-15)
        return float(np.max(relative_error))


def run_symbolic_compression_test():
    """Run symbolic compression test and return results for analysis."""
    print("🧮 TESTING SYMBOLIC MATHEMATICAL COMPRESSION")
    print("=" * 60)

    compressor = SymbolicMathematicalCompressor()

    test_cases = [
        ("Mathematical Constants", np.array([np.pi] * 1000 + [np.e] * 500 + [np.sqrt(2)] * 300)),
        ("Arithmetic Progression", np.linspace(0, 100, 2000)),
        ("Geometric Progression", np.array([2.0**i for i in range(20)] * 100)),
        ("Polynomial Pattern", np.array([i**2 + 2 * i + 1 for i in range(1000)])),
        ("Repetitive Pattern", np.tile(np.array([1.1, 2.2, 3.3, 4.4, 5.5]), 400)),
        ("Mixed Mathematical", np.concatenate([
            np.array([np.pi] * 100), np.linspace(1, 10, 200),
            np.array([np.e] * 50), np.array([1.618] * 75),
        ])),
    ]

    total_original_size = 0
    total_compressed_size = 0
    results = []

    for test_name, test_data in test_cases:
        print(f"\n🔍 Testing: {test_name} ({test_data.size} elements)")

        compressed_data, stats = compressor.compress_array_symbolic(test_data)

        try:
            decompressed_data, decomp_stats = compressor.decompress_array_symbolic(compressed_data)

            if test_data.shape == decompressed_data.shape:
                max_error = np.max(np.abs(test_data - decompressed_data))
                accuracy_good = max_error < 1e-12
            else:
                max_error = float("inf")
                accuracy_good = False
            print(f"   Compression: {stats['compression_ratio']:.2f}x, Patterns: {stats['patterns_detected']}, Error: {max_error:.2e}, Status: {'✅ GOOD' if accuracy_good else '❌ POOR'}")

            total_original_size += stats["original_size"]
            total_compressed_size += stats["compressed_size"]

            results.append({
                "name": test_name, "compression_ratio": stats["compression_ratio"],
                "patterns": stats["patterns_detected"], "accuracy": accuracy_good, "max_error": max_error,
            })

        except Exception as e:
            print(f"   ❌ Decompression failed: {e}")
            results.append({"name": test_name, "compression_ratio": 0, "patterns": 0, "accuracy": False, "max_error": float("inf")})

    overall_ratio = (
        total_original_size / total_compressed_size
        if total_compressed_size > 0
        else 0
    )
    successful_tests = sum(1 for r in results if r["accuracy"])

    print("\n📊 OVERALL SYMBOLIC COMPRESSION RESULTS")
    print("=" * 50)
    print(f"Overall compression ratio: {overall_ratio:.2f}x")
    print(f"Successful tests: {successful_tests}/{len(results)}")
    print(
        f"Target achieved (2.0x): {
            '✅ YES' if overall_ratio >= 2.0 else '❌ NO'}"
    )

    best_ratios = sorted(
        results, key=lambda x: x["compression_ratio"], reverse=True
    )[:3]
    print("\nBest compression ratios:")
    for i, result in enumerate(best_ratios, 1):
        print(f"  {i}. {result['name']}: {result['compression_ratio']:.2f}x")

    return {
        "overall_compression_ratio": overall_ratio,
        "successful_tests": successful_tests,
        "total_tests": len(results),
        "target_achieved": overall_ratio >= 2.0,
        "results": results,
    }


def test_symbolic_compression():
    """Pytest-compatible test function for symbolic compression."""
    results = run_symbolic_compression_test()

    assert results[
        "target_achieved"
    ], f"Compression target not met: {
        results['overall_compression_ratio']:.2f}x < 2.0x required"
    assert (
        results["successful_tests"] == results["total_tests"]
    ), f"Some accuracy tests failed: {results['successful_tests']}/{results['total_tests']}"


if __name__ == "__main__":
    print("🔬 SYMBOLIC MATHEMATICAL COMPRESSION SYSTEM")
    print("=" * 80)
    print("Advanced compression for dimensional mathematics data")
    print("TARGET: Achieve 2.0x+ compression ratio with precision preservation")
    try:
        test_results = run_symbolic_compression_test()
        if test_results["target_achieved"]:
            print("\n🎉 SYMBOLIC COMPRESSION SUCCESS!")
            print("2.0x+ compression target achieved with precision preservation")
        else:
            print(f"\n⚠️ Need improvement: {test_results['overall_compression_ratio']:.2f}x current vs 2.0x target")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
