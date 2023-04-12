use clap::Parser;
use kdam::tqdm;
use nalgebra::Matrix4;
use nifti::IntoNdArray;
use nifti::{writer::WriterOptions, NiftiObject, ReaderOptions};
use nifti_processing::{resample_to_output, NearestNeighbor, ReSample, TriLinear};
use std::path::Path;

#[derive(Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Input nifti files
    #[arg(required = false, value_name = "INPUT")]
    inputs: Vec<String>,

    /// Resolution
    #[arg(short, default_value_t = 1.5f32)]
    resolution: f32,

    /// Order
    #[arg(short='n', default_value_t = 1i8)]
    order: i8,

    /// Output directory
    #[arg(short, default_value_t = String::from("./"))]
    output_directory: String,
}

#[allow(unused_variables)]
fn main() {
    let args = Args::parse();

    let output_dir = Path::new(&args.output_directory);

    let sampler_nn = NearestNeighbor::default();
    let sampler_tri = TriLinear::default();

    for filename in tqdm!(args.inputs.iter()) {
        let path = Path::new(filename);
        let output_path = output_dir.join(path.file_name().unwrap());
        let in_nii = match ReaderOptions::new().read_file(path) {
            Ok(nii) => nii,
            Err(_err) => {
                println!("failed to load: {filename}");
                continue;
            }
        };

        let in_header = in_nii.header();
        let in_affine: Matrix4<f32> = in_header.affine();
        let in_volume = in_nii.volume();
        let in_im = match in_volume.into_ndarray::<f32>() {
            Ok(im) => im,
            Err(_err) => {
                println!("failed to load im array: {filename}");
                continue;
            }
        };

        let sampler: &dyn ReSample<f32, f32> = match args.order {
            0 => &sampler_nn,
            1 => &sampler_tri,
            _ => panic!("invalid order argument"),
        };

        let (out_im, out_affine) =
            match resample_to_output(&in_im, &in_affine, &[args.resolution; 3], sampler) {
                Ok(r) => r,
                Err(_err) => {
                    println!("failed to process: {filename}");
                    println!("{_err}");
                    continue;
                }
            };
        let mut out_header = in_header.clone();
        out_header.set_affine(&out_affine);

        match WriterOptions::new(output_path)
            .reference_header(&out_header)
            .write_nifti(&out_im)
        {
            Ok(r) => r,
            Err(_err) => {
                println!("failed to write: {filename}");
                continue;
            }
        }
    }
}
