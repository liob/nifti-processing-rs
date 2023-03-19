use super::common::SamplingMode;
use super::traits::ReSample;
use nalgebra::{MatrixXx3, RealField};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};

/// A sampler employing a trilinear interpolation strategy.
///
/// This sampler corresponds to `order=1` in nibabel.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriLinear<U>
where
    U: Num + Copy,
{
    mode: SamplingMode,
    cval: U,
}

impl<U> Default for TriLinear<U>
where
    U: Num + Copy,
{
    fn default() -> Self {
        Self {
            mode: SamplingMode::Constant,
            cval: U::zero(),
        }
    }
}

impl<T, U> ReSample<T, U> for TriLinear<U>
where
    T: Num + AsPrimitive<usize> + RealField + PartialOrd + Copy,
    U: Num + Copy,
    usize: AsPrimitive<T>,
{
    fn set_sampling_mode(&mut self, mode: SamplingMode) {
        self.mode = mode;
    }

    fn get_sampling_mode(&self) -> SamplingMode {
        self.mode
    }

    fn set_cval(&mut self, cval: U) {
        self.cval = cval;
    }

    fn get_cval(&self) -> U {
        self.cval
    }

    fn sample(
        &self,
        in_im: &Array<U, IxDyn>,
        in_coords: &mut MatrixXx3<T>,
        out_shape: &[usize],
    ) -> Result<Array<U, IxDyn>, String> {
        let mut values: Vec<U> = Vec::with_capacity(in_coords.len());
        self.apply_sampling_mode(in_im, in_coords);

        let in_coords_0 =
            MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter().map(|x| x.floor()));
        let in_coords_1 =
            MatrixXx3::from_iterator(in_coords_0.nrows(), in_coords_0.iter().map(|x| *x + T::one()));

        for i in 0..in_coords.nrows() {
            
        }


        if let Ok(r) = Array::from_shape_vec(out_shape, values) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
