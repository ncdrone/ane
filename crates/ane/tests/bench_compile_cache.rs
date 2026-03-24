use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ane::{Graph, NSQualityOfService, Shape};

const CHILD_ENV: &str = "ANE_COMPILE_CACHE_CHILD";
const MODEL_ARTIFACT_DIR_ENV: &str = "ANE_MODEL_ARTIFACT_DIR";
const CHILD_METRIC_PREFIX: &str = "ANE_COMPILE_CACHE_MS=";

type TestResult = Result<(), Box<dyn Error>>;

fn deterministic_values(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - (modulus / 2) as f32) * scale)
        .collect()
}

fn build_compile_graph() -> Graph {
    let channels = 768usize;
    let width = 512usize;
    let hidden = 768usize;

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

    let w1 = g.constant(
        &deterministic_values(channels * hidden, 17, 0.01),
        Shape {
            batch: 1,
            channels: 1,
            height: channels,
            width: hidden,
        },
    );
    let mm1 = g.matrix_multiplication(input_t, w1, false, false);
    let mm1_t = g.transpose(mm1, [0, 1, 3, 2]);
    let hidden_1 = g.reshape(
        mm1_t,
        Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width,
        },
    );
    let relu = g.relu(hidden_1);

    let relu_2d = g.reshape(
        relu,
        Shape {
            batch: 1,
            channels: 1,
            height: hidden,
            width,
        },
    );
    let relu_t = g.transpose(relu_2d, [0, 1, 3, 2]);
    let w2 = g.constant(
        &deterministic_values(hidden * channels, 23, 0.008),
        Shape {
            batch: 1,
            channels: 1,
            height: hidden,
            width: channels,
        },
    );
    let mm2 = g.matrix_multiplication(relu_t, w2, false, false);
    let mm2_t = g.transpose(mm2, [0, 1, 3, 2]);
    let _output = g.reshape(
        mm2_t,
        Shape {
            batch: 1,
            channels,
            height: 1,
            width,
        },
    );

    g
}

fn compile_once_ms() -> Result<f64, ane::Error> {
    let graph = build_compile_graph();
    let start = Instant::now();
    let executable = graph.compile(NSQualityOfService::Default)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    drop(executable);
    Ok(elapsed_ms)
}

fn parse_metric(output: &[u8]) -> Result<f64, Box<dyn Error>> {
    let text = String::from_utf8(output.to_vec())?;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix(CHILD_METRIC_PREFIX) {
            return Ok(value.trim().parse()?);
        }
    }
    Err(format!("missing metric line in child output:\n{text}").into())
}

fn run_child(model_artifact_dir: Option<&Path>) -> Result<f64, Box<dyn Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("-p")
        .arg("ane")
        .arg("--test")
        .arg("bench_compile_cache")
        .arg("compile_cache_child_entrypoint")
        .arg("--")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1");

    if let Some(path) = model_artifact_dir {
        cmd.env(MODEL_ARTIFACT_DIR_ENV, path);
    } else {
        cmd.env_remove(MODEL_ARTIFACT_DIR_ENV);
    }

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "child benchmark failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    parse_metric(&output.stdout)
}

fn clear_dir(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}

#[test]
fn compile_cache_child_entrypoint() -> TestResult {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }

    let elapsed_ms = compile_once_ms()?;
    println!("{CHILD_METRIC_PREFIX}{elapsed_ms:.6}");
    Ok(())
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture on ANE hardware"]
fn bench_compile_cache_same_process() -> TestResult {
    let first_ms = compile_once_ms()?;
    let second_ms = compile_once_ms()?;
    let speedup = if second_ms > 0.0 {
        first_ms / second_ms
    } else {
        f64::INFINITY
    };

    println!(
        "same_process_compile_cache first_ms={first_ms:.3} second_ms={second_ms:.3} speedup={speedup:.2}x"
    );
    assert!(first_ms > 0.0);
    assert!(second_ms > 0.0);
    Ok(())
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture on ANE hardware"]
fn bench_compile_cache_cross_process() -> TestResult {
    let fixed_dir: PathBuf = std::env::temp_dir().join("ane-bench-compile-cache-fixed-path");

    clear_dir(&fixed_dir)?;
    let default_first_ms = run_child(None)?;
    let default_second_ms = run_child(None)?;
    let default_speedup = if default_second_ms > 0.0 {
        default_first_ms / default_second_ms
    } else {
        f64::INFINITY
    };

    println!(
        "cross_process_compile_cache default_first_ms={default_first_ms:.3} default_second_ms={default_second_ms:.3} default_speedup={default_speedup:.2}x"
    );

    assert!(default_first_ms > 0.0);
    assert!(default_second_ms > 0.0);

    clear_dir(&fixed_dir)?;
    match (run_child(Some(&fixed_dir)), run_child(Some(&fixed_dir))) {
        (Ok(fixed_first_ms), Ok(fixed_second_ms)) => {
            let fixed_speedup = if fixed_second_ms > 0.0 {
                fixed_first_ms / fixed_second_ms
            } else {
                f64::INFINITY
            };
            println!(
                "cross_process_compile_cache fixed_first_ms={fixed_first_ms:.3} fixed_second_ms={fixed_second_ms:.3} fixed_speedup={fixed_speedup:.2}x fixed_dir={}",
                fixed_dir.display(),
            );
            assert!(fixed_first_ms > 0.0);
            assert!(fixed_second_ms > 0.0);
        }
        (first, second) => {
            println!(
                "cross_process_compile_cache fixed_path_unavailable first={first:?} second={second:?} fixed_dir={}",
                fixed_dir.display(),
            );
        }
    }

    Ok(())
}
