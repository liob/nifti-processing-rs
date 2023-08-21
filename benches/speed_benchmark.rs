use ndarray;
use nalgebra::MatrixXx3;
use nifti_processing::{TriLinear, NearestNeighbor, ReSample};
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    Criterion
};

fn sampling_benchmark(c: &mut Criterion) {
    let in_im = ndarray::Array::range(0., 1000000., 1.);
    let in_im = in_im.into_shape((100, 100, 100)).unwrap();
    let in_im = in_im.mapv(|x| f32::from(x)).into_dyn();

    let mut in_coords: Vec<f32> = Vec::default();
    for x in 0..100 {
        for y in 0..100 {
            for z in 0..100 {
                in_coords.extend_from_slice(&[x as f32, y as f32, z as f32])
            }
        }
    };
    let mut in_coords = MatrixXx3::from_row_slice(&in_coords);

    let sampler = NearestNeighbor::<f32>::default();
    c.bench_function(
        "nearest neighbor resampling", 
        |b| b.iter(|| sampler.sample(&in_im, &mut in_coords, &[100, 100, 100]))
    );

    let sampler = TriLinear::<f32>::default();
    c.bench_function(
        "trilinear resampling", 
        |b| b.iter(|| sampler.sample(&in_im, &mut in_coords, &[100, 100, 100]))
    );
}

criterion_group!(benches, sampling_benchmark);
criterion_main!(benches);
