use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, ChiSquared, Exp, InverseGaussian, Normal, Pareto};

use std::f64::INFINITY;
const EPSILON: f64 = 1e-8;
use std::f64::consts::PI;

struct Job {
    rem_size: f64,
    arrival_time: f64,
}

//Only insertion policies
#[derive(Debug)]
enum Policy {
    SRPT,
    SPT,
    FCFS,
    PLCFS,
    Nudge(f64, bool),
}
impl Policy {
    fn insertion_index(&mut self, queue: &Vec<Job>, new_job: &Job) -> usize {
        match self {
            Policy::SRPT => {
                let wrapped_index =
                    queue.binary_search_by_key(&n64(new_job.rem_size), |job| n64(job.rem_size));
                match wrapped_index {
                    Ok(ok) => ok,
                    Err(err) => err,
                }
            }
            Policy::SPT => {
                if queue.is_empty() {
                    0
                } else {
                    let wrapped_index =
                        queue.binary_search_by_key(&n64(new_job.rem_size), |job| n64(job.rem_size));
                    let index = match wrapped_index {
                        Ok(ok) => ok,
                        Err(err) => err,
                    };
                    index.max(1)
                }
            }
            Policy::FCFS => queue.len(),
            Policy::PLCFS => 0,
            Policy::Nudge(threshold, mut just_swapped) => {
                if !just_swapped {
                    if queue.len() > 1 {
                        let last_job = queue.last().unwrap();
                        if last_job.rem_size > *threshold && new_job.rem_size <= *threshold {
                            just_swapped = true;
                            queue.len() - 1
                        } else {
                            queue.len()
                        }
                    } else {
                        queue.len()
                    }
                } else {
                    just_swapped = false;
                    queue.len()
                }
            }
        }
    }
}

struct Results {
    response_times: Vec<usize>,
    step: f64,
}

fn simulate(
    lambda: f64,
    dist: Dist,
    policy: &mut Policy,
    step: f64,
    num_jobs: usize,
    seed: u64,
) -> Results {
    assert!((dist.mean() - 1.0).abs() < EPSILON);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut time = 0.0;
    let arrival_dist = Exp::new(lambda).unwrap();
    let mut next_arrival = rng.sample(arrival_dist);
    let mut num_completions = 0;
    let mut queue: Vec<Job> = vec![];
    let mut results = Results {
        step,
        response_times: vec![],
    };
    while num_completions < num_jobs {
        let next_event_diff =
            (next_arrival - time).min(queue.first().map_or(INFINITY, |j| j.rem_size));
        let was_arrival = next_event_diff == (next_arrival - time);
        time += next_event_diff;
        if !queue.is_empty() {
            let job = queue.first_mut().unwrap();
            job.rem_size -= next_event_diff;
            if job.rem_size <= EPSILON {
                let job = queue.remove(0);
                let response = time - job.arrival_time;
                let response_index = (response / step) as usize;
                while results.response_times.len() <= response_index {
                    results.response_times.push(0)
                }
                results.response_times[response_index] += 1;
                num_completions += 1;
            }
        }
        if was_arrival {
            let size = dist.sample(&mut rng);
            let new_job = Job {
                rem_size: size,
                arrival_time: time,
            };
            next_arrival = time + arrival_dist.sample(&mut rng);
            let insertion_index = policy.insertion_index(&queue, &new_job);
            queue.insert(insertion_index, new_job);
        }
    }
    results
}

#[derive(Clone, Copy, Debug)]
enum Dist {
    Uniform(f64, f64),
    Exponential(f64),
    Hyperexponential(f64, f64, f64),
    ShiftedBoundedPareto(f64, f64, f64),
    Beta(f64, f64, f64),
    ChiSquared(f64),
    InverseGaussian(f64, f64),
    MixedUniform(f64, f64, f64),
    HalfNormal(f64),
    Triangle(f64, f64),
    Erlang(u64, f64),
    InfiniteDensity,
    Lomax(f64, f64),
}

impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let sample = match self {
            Dist::Uniform(low, high) => rng.gen_range(*low..*high),
            Dist::Exponential(mean) => rng.sample(Exp::new(1.0 / mean).unwrap()),
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                let mean = if rng.gen::<f64>() < *prob_low {
                    low_mean
                } else {
                    high_mean
                };
                rng.sample(Exp::new(1.0 / mean).unwrap())
            }
            Dist::ShiftedBoundedPareto(min, max, alpha) => {
                let pareto = Pareto::new(*min, *alpha).unwrap();
                loop {
                    let sample = rng.sample(pareto);
                    if sample <= *max {
                        break sample - min;
                    }
                }
            }
            Dist::Beta(alpha, beta, scale) => {
                let beta = Beta::new(*alpha, *beta).unwrap();
                rng.sample(beta) * scale
            }
            Dist::ChiSquared(k) => {
                let chi_squared = ChiSquared::new(*k).unwrap();
                rng.sample(chi_squared)
            }
            Dist::InverseGaussian(mean, shape) => {
                let inv_gaussian = InverseGaussian::new(*mean, *shape).unwrap();
                rng.sample(inv_gaussian)
            }
            Dist::MixedUniform(low_upper, high_upper, prob_low) => {
                let upper = if rng.gen::<f64>() < *prob_low {
                    low_upper
                } else {
                    high_upper
                };
                rng.gen_range(0.0..*upper)
            }
            Dist::HalfNormal(var) => {
                let normal = Normal::new(0.0, *var).unwrap();
                rng.sample(normal).abs()
            }
            Dist::Triangle(low, high) => {
                let dist_from_center = rng.gen_range(0.0..(high - low) / 2.0);
                let low_prob = 0.5 + dist_from_center / (high - low);
                if rng.gen::<f64>() < low_prob {
                    (low + high) / 2.0 - dist_from_center
                } else {
                    (low + high) / 2.0 + dist_from_center
                }
            }
            Dist::Erlang(k, mean) => {
                let exp = Exp::new(1.0 / mean).unwrap();
                (0..*k).map(|_| rng.sample(exp)).sum()
            }
            // pdf: 1/4 * |x-1|^{0.5}
            Dist::InfiniteDensity => {
                let sample = rng.gen_range(0.0..1.0);
                1.0 + (2.0 * sample - 1.0f64).powi(2) * (sample - 0.5f64).signum()
            }
            Dist::Lomax(lambda, alpha) => {
                rng.sample(Pareto::new(*lambda, *alpha).unwrap()) - lambda
            }
        };
        assert!(sample >= 0.0);
        sample
    }
    fn mean(&self) -> f64 {
        match self {
            Dist::Uniform(low, high) => (low + high) / 2.0,
            Dist::Exponential(mean) => *mean,
            Dist::Hyperexponential(low_mean, high_mean, prob_low) => {
                low_mean * prob_low + high_mean * (1.0 - prob_low)
            }
            Dist::ShiftedBoundedPareto(min, max, alpha) => {
                min.powf(*alpha) / (1.0 - (min / max).powf(*alpha))
                    * (alpha / (alpha - 1.0))
                    * (1.0 / min.powf(alpha - 1.0) - 1.0 / max.powf(alpha - 1.0))
                    - min
            }
            Dist::Beta(alpha, beta, scale) => scale * alpha / (alpha + beta),
            Dist::ChiSquared(k) => *k,
            Dist::InverseGaussian(mean, _shape) => *mean,
            Dist::MixedUniform(low_upper, high_upper, prob_low) => {
                (low_upper * prob_low + high_upper * (1.0 - prob_low)) / 2.0
            }
            Dist::HalfNormal(var) => (var * 2.0 / PI).sqrt(),
            Dist::Triangle(low, high) => (2.0 * low + high) / 3.0,
            Dist::Erlang(k, mean) => *k as f64 * mean,
            Dist::InfiniteDensity => 1.0,
            Dist::Lomax(lambda, alpha) => lambda / (alpha - 1.0),
        }
    }
}

fn main() {
    let num_jobs = 2_000_000_000;
    let rho = 0.4;
    let seed = 0;
    let step = 0.1;
    let dist = Dist::Hyperexponential(0.5, 3.0, 0.8);
    let policies = vec![
        Policy::SRPT,
        Policy::SPT,
        Policy::FCFS,
        Policy::PLCFS,
        Policy::Nudge(1.0, false),
    ];
    for mut policy in policies {
        let results = simulate(rho, dist, &mut policy, step, num_jobs, seed);
        assert_eq!(results.response_times.iter().sum::<usize>(), num_jobs);
        let cumulant: Vec<usize> = results
            .response_times
            .iter()
            .scan(num_jobs, |state, &count| {
                *state -= count;
                Some(*state)
            })
            .collect();
        let log_frequencies = cumulant
            .iter()
            .take((100.0/step) as usize)
            .map(|&freq| (freq as f64 / num_jobs as f64).log10());
        println!(
            "{:?};{}",
            policy,
            log_frequencies
                .map(|f| format!("{}", f))
                .collect::<Vec<String>>()
                .join(";")
        );
    }
}
