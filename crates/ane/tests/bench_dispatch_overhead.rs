use std::error::Error;
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};

type TestResult = Result<(), Box<dyn Error>>;

fn deterministic_values(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - (modulus / 2) as f32) * scale)
        .collect()
}

fn build_projection_graph(channels: usize, width: usize, out_channels: usize) -> (Graph, Shape) {
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(channels, 1, width));
    let input_2d = g.reshape(
        input,
        Shape {
            batch: 1,
            channels: 1,
            height: channels,
            width,
        },
    );
    let input_t = g.transpose(input_2d, [0, 1, 3, 2]);

    let weights = g.constant(
        &deterministic_values(channels * out_channels, 19, 0.01),
        Shape {
            batch: 1,
            channels: 1,
            height: channels,
            width: out_channels,
        },
    );
    let mm = g.matrix_multiplication(input_t, weights, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    let _output = g.reshape(
        mm_t,
        Shape {
            batch: 1,
            channels: out_channels,
            height: 1,
            width,
        },
    );

    (g, Shape::spatial(out_channels, 1, width))
}

fn compile_projection_executable(
    channels: usize,
    width: usize,
    out_channels: usize,
) -> Result<(Executable, TensorData, TensorData), ane::Error> {
    let (graph, output_shape) = build_projection_graph(channels, width, out_channels);
    let executable = graph.compile(NSQualityOfService::Default)?;
    let input = TensorData::with_f32(
        &deterministic_values(channels * width, 29, 0.02),
        Shape::spatial(channels, 1, width),
    );
    let output = TensorData::new(output_shape);
    Ok((executable, input, output))
}

fn bench_dispatch<F>(label: &str, iters: usize, mut run: F) -> Result<f64, ane::Error>
where
    F: FnMut() -> Result<(), ane::Error>,
{
    for _ in 0..50 {
        run()?;
    }

    let samples = 5usize;
    let iters_per_sample = (iters / samples).max(1);
    let mut sample_us = Vec::with_capacity(samples);

    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iters_per_sample {
            run()?;
        }
        sample_us.push(start.elapsed().as_secs_f64() * 1e6 / iters_per_sample as f64);
    }

    sample_us.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let median_us = sample_us[samples / 2];
    println!("{label} sample_per_dispatch_us={sample_us:?} median_per_dispatch_us={median_us:.3}");
    Ok(median_us)
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture on ANE hardware"]
fn bench_dispatch_overhead_cached_vs_direct_vs_premap() -> TestResult {
    let channels = 768usize;
    let width = 512usize;
    let out_channels = 768usize;
    let iters = 1000usize;

    let (cached_exec, cached_input, cached_output) =
        compile_projection_executable(channels, width, out_channels)?;
    let cached_us = bench_dispatch("run_cached", iters, || {
        cached_exec.run_cached(&[&cached_input], &[&cached_output])
    })?;

    let (direct_exec, direct_input, direct_output) =
        compile_projection_executable(channels, width, out_channels)?;
    let direct_us = bench_dispatch("run_cached_direct", iters, || {
        direct_exec.run_cached_direct(&[&direct_input], &[&direct_output])
    })?;

    let (premap_exec, premap_input, premap_output) =
        compile_projection_executable(channels, width, out_channels)?;
    let premap_us = match premap_exec.pre_map_request(&[&premap_input], &[&premap_output]) {
        Ok(()) => Some(bench_dispatch("run_cached_after_premap", iters, || {
            premap_exec.run_cached(&[&premap_input], &[&premap_output])
        })?),
        Err(error) => {
            println!("run_cached_after_premap unavailable error={error}");
            None
        }
    };

    println!(
        "dispatch_overhead_summary cached_us={cached_us:.3} direct_us={direct_us:.3} premap_us={} direct_minus_cached_us={:.3}{}",
        premap_us
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".into()),
        direct_us - cached_us,
        premap_us
            .map(|value| format!(" premap_minus_cached_us={:.3}", value - cached_us))
            .unwrap_or_default(),
    );

    assert!(cached_us > 0.0);
    assert!(direct_us > 0.0);
    if let Some(premap_us) = premap_us {
        assert!(premap_us > 0.0);
    }
    Ok(())
}

#[test]
#[ignore = "requires ANE hardware"]
fn perf_stats_reports_non_zero_hw_execution_time() -> TestResult {
    let channels = 768usize;
    let width = 512usize;
    let out_channels = 768usize;

    let (executable, input, output) = compile_projection_executable(channels, width, out_channels)?;

    let mut last_hw_time = 0u64;
    for _ in 0..10 {
        last_hw_time = executable.run_cached_with_stats(&[&input], &[&output])?;
        if last_hw_time > 0 {
            break;
        }
    }

    println!("perf_stats hw_execution_time_ns={last_hw_time}");
    assert!(
        last_hw_time > 0,
        "expected non-zero hw_execution_time from run_cached_with_stats",
    );
    Ok(())
}
