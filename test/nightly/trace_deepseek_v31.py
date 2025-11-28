"""Trace DeepSeek V3.1 performance with different TP sizes and MoE backends."""

import unittest

from nightly_utils import NightlyBenchmarkRunner
from trace_results_manager import TraceResultsManager

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

# Shared profile directory for all trace tests
PROFILE_DIR = "performance_profiles_neurips_traces"

MODEL_NAME = "DeepSeek-V3.1"

MODEL_CONFIG = {
    "path": "deepseek-ai/DeepSeek-V3.1",
    "extra_args": [
        "--trust-remote-code",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
    ],
}

BATCH_SIZES = [1]
INPUT_LENS = (4096,)
OUTPUT_LENS = (512,)

# Test configurations: (tp_size, ep_size, quant_backend_pairs)
# Each entry: (tp_size, ep_size, [(quantization, moe_backend, fallback_quantization, fallback_backend), ...])
CONFIGS = [
    (
        4,
        1,
        [
            ("fp8", "flashinfer_trtllm", None, None),
            ("auto", "triton", None, None),
            ("auto", None, None, None),
        ],
    ),  # TP4, EP1 - likely OOM
    (
        8,
        1,
        [
            ("fp8", "flashinfer_trtllm", None, None),
            ("auto", "triton", None, None),
            ("auto", None, None, None),
        ],
    ),  # TP8, EP1 - may fail if expert count not divisible
    (
        8,
        2,
        [
            ("fp8", "flashinfer_trtllm", None, None),
            ("auto", "triton", None, None),
            ("auto", None, None, None),
        ],
    ),  # TP8, EP2 - should work
]


class TraceDeepSeekV31(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.runner = NightlyBenchmarkRunner(
            PROFILE_DIR, "DeepSeek V3.1 Performance Traces", cls.base_url
        )
        cls.runner.setup_profile_directory()
        cls.results_manager = TraceResultsManager(PROFILE_DIR)

    def test_deepseek_v31_configs(self):
        """Test DeepSeek V3.1 with different TP/EP/MoE configurations."""

        all_results = []
        successful_configs = []
        failed_configs = []

        for tp_size, ep_size, quant_backend_pairs in CONFIGS:
            for (
                quant,
                moe_backend,
                fallback_quant,
                fallback_backend,
            ) in quant_backend_pairs:
                # Try primary configuration
                server_args = ["--tp", str(tp_size)] + MODEL_CONFIG["extra_args"]

                if ep_size > 1:
                    server_args += ["--ep", str(ep_size)]

                if moe_backend is not None:
                    server_args += ["--moe-runner-backend", moe_backend]

                if quant == "fp8":
                    server_args += ["--quantization", "fp8"]

                ep_str = f"_EP{ep_size}" if ep_size > 1 else ""
                quant_str = f"{quant}_" if quant != "auto" else ""
                backend_str = moe_backend if moe_backend else "default"
                variant = f"TP{tp_size}{ep_str}_{quant_str}{backend_str}"
                config_name = f"deepseek-v3.1 {variant}"

                print(f"\n{'='*80}")
                print(f"Running {config_name}...")
                print(f"{'='*80}\n")

                primary_success = False
                try:
                    results, success = self.runner.run_benchmark_for_model(
                        model_path=MODEL_CONFIG["path"],
                        batch_sizes=BATCH_SIZES,
                        input_lens=INPUT_LENS,
                        output_lens=OUTPUT_LENS,
                        other_args=server_args,
                        variant=variant,
                        extra_bench_args=["--trust-remote-code"],
                    )

                    if success and results:
                        all_results.extend(results)
                        self.runner.add_report(results)
                        successful_configs.append(config_name)
                        print(f"✓ Success: {config_name}")
                        primary_success = True

                        # Record success in results manager
                        for result in results:
                            # Calculate inter-token latency from output throughput
                            itl_ms = (
                                1
                                / (result.output_throughput / result.batch_size)
                                * 1000
                                if result.output_throughput > 0
                                else None
                            )
                            self.results_manager.record_success(
                                model=MODEL_NAME,
                                config=variant,
                                batch_size=result.batch_size,
                                input_len=result.input_len,
                                output_len=result.output_len,
                                latency_s=result.latency,
                                input_throughput=result.input_throughput,
                                output_throughput=result.output_throughput,
                                itl_ms=itl_ms,
                            )
                    else:
                        print(f"⚠️  Primary config failed: {config_name}")
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️  Primary config error: {error_msg}")

                # If primary failed and fallback is specified, try fallback
                if not primary_success and fallback_backend is not None:
                    fallback_server_args = ["--tp", str(tp_size)] + MODEL_CONFIG[
                        "extra_args"
                    ]

                    if ep_size > 1:
                        fallback_server_args += ["--ep", str(ep_size)]

                    if fallback_backend is not None:
                        fallback_server_args += [
                            "--moe-runner-backend",
                            fallback_backend,
                        ]

                    if fallback_quant == "fp8":
                        fallback_server_args += ["--quantization", "fp8"]

                    fallback_quant_str = (
                        f"{fallback_quant}_"
                        if fallback_quant and fallback_quant != "auto"
                        else ""
                    )
                    fallback_backend_str = (
                        fallback_backend if fallback_backend else "default"
                    )
                    fallback_variant = f"TP{tp_size}{ep_str}_{fallback_quant_str}{fallback_backend_str}"
                    fallback_config_name = f"deepseek-v3.1 {fallback_variant}"

                    print(f"\n{'='*80}")
                    print(f"Trying fallback: {fallback_config_name}...")
                    print(f"{'='*80}\n")

                    try:
                        results, success = self.runner.run_benchmark_for_model(
                            model_path=MODEL_CONFIG["path"],
                            batch_sizes=BATCH_SIZES,
                            input_lens=INPUT_LENS,
                            output_lens=OUTPUT_LENS,
                            other_args=fallback_server_args,
                            variant=fallback_variant,
                            extra_bench_args=[],
                        )

                        if success and results:
                            all_results.extend(results)
                            self.runner.add_report(results)
                            successful_configs.append(fallback_config_name)
                            print(f"✓ Success (fallback): {fallback_config_name}")

                            # Record success in results manager
                            for result in results:
                                # Calculate inter-token latency from output throughput
                                itl_ms = (
                                    1
                                    / (result.output_throughput / result.batch_size)
                                    * 1000
                                    if result.output_throughput > 0
                                    else None
                                )
                                self.results_manager.record_success(
                                    model=MODEL_NAME,
                                    config=fallback_variant,
                                    batch_size=result.batch_size,
                                    input_len=result.input_len,
                                    output_len=result.output_len,
                                    latency_s=result.latency,
                                    input_throughput=result.input_throughput,
                                    output_throughput=result.output_throughput,
                                    itl_ms=itl_ms,
                                )
                        else:
                            failed_configs.append(fallback_config_name)
                            print(f"⚠️  Failed (fallback): {fallback_config_name}")
                            # Record failure
                            self.results_manager.record_failure(
                                model=MODEL_NAME,
                                config=fallback_variant,
                                error_message="Benchmark failed to complete",
                                error_type="BenchmarkFailure",
                            )
                    except Exception as e:
                        failed_configs.append(fallback_config_name)
                        error_msg = str(e)
                        print(f"⚠️  Error (fallback): {error_msg}")

                        # Determine error type
                        error_type = "Unknown"
                        if (
                            "out of memory" in error_msg.lower()
                            or "oom" in error_msg.lower()
                        ):
                            error_type = "OOM"
                        elif "timeout" in error_msg.lower():
                            error_type = "Timeout"
                        elif "server failed" in error_msg.lower():
                            error_type = "ServerError"

                        # Record failure
                        self.results_manager.record_failure(
                            model=MODEL_NAME,
                            config=fallback_variant,
                            error_message=error_msg,
                            error_type=error_type,
                        )
                elif not primary_success:
                    # No fallback, record primary failure
                    failed_configs.append(config_name)
                    self.results_manager.record_failure(
                        model=MODEL_NAME,
                        config=variant,
                        error_message="Benchmark failed to complete",
                        error_type="BenchmarkFailure",
                    )

        # Write final report
        self.runner.write_final_report()

        # Print results manager summary
        self.results_manager.print_summary()

        print(f"\n{'='*80}")
        print(f"THIS RUN SUMMARY")
        print(f"{'='*80}")
        print(f"Successful: {len(successful_configs)}")
        print(f"Failed: {len(failed_configs)}")

        if successful_configs:
            print(f"\n✓ Successful:")
            for config in successful_configs:
                print(f"  - {config}")

        # Print markdown table
        print(f"\n{self.results_manager.generate_markdown_table()}")

        if failed_configs:
            print(f"\n⚠️  Failed:")
            for config in failed_configs:
                print(f"  - {config}")


if __name__ == "__main__":
    unittest.main()
