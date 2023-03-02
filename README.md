# NIFTI-Processing-rs
This library is an extension of the [NIFTI-rs] library, adding resampling support. This library is closely modeled after the [NiBabel] [processing module][NiBabel-processing], hence the name.


## Features
The `resample_to_output` and `resample_from_to` functions are implemented.


## Limitations
  - Only nearest neighbor interpolation is supported. Trilinear interpolation will follow.
  - Minimal error checking; Will be extended.
  - No unit tests.
  - Please also consult the issue tracker.


## Requirements
The `nalgebra_affine` and `ndarray_volumes` features of NIFTI-rs are required.


## Example
```rust
use nifti::{NiftiObject, ReaderOptions, NiftiVolume};
use use nifti_processing::{resample_to_output, sampler};

let obj = ReaderOptions::new().read_file("myvolume.nii.gz")?;
let header = obj.header();
let affine = header.get_affine();
let volume = obj.volume();
let im = volume.into_ndarray::<f32>()?;

let nn = sampler::NearestNeighbor::default();
let (resampled_im, resampled_affine) = resample_to_output(&im, &affine, &[1.0,1.0,1.0], nn)
```

See also the examples directory.


## License
Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any Contribution intentionally submitted for inclusion in work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.



[NiBabel]: https://nipy.org/nibabel/
[NiBabel-processing]: https://nipy.org/nibabel/reference/nibabel.processing.html
[NIFTI-rs]: https://github.com/Enet4/nifti-rs
