use super::common::SamplingMode;
use super::traits::ReSample;
use nalgebra::{MatrixXx3, RealField};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};

/// A sampler employing a nearest neighbor strategy.
///
/// This sampler corresponds to `order=0` in nibabel.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestNeighbor<U>
where
    U: Num + Copy,
{
    mode: SamplingMode,
    cval: U,
}

impl<U> Default for NearestNeighbor<U>
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

impl<T, U> ReSample<T, U> for NearestNeighbor<U>
where
    T: Num + AsPrimitive<usize> + AsPrimitive<U> + RealField + PartialOrd + Copy,
    U: Num + Copy + 'static,
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

        let mut in_coords =
            MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter_mut().map(|x| x.ceil()));
        self.apply_sampling_mode(in_im, &mut in_coords);

        for in_coord in in_coords.row_iter() {
            let (x, y, z) = (in_coord[(0, 0)], in_coord[(0, 1)], in_coord[(0, 2)]);
            values.push(self.get_val(in_im, x, y, z));
        }

        if let Ok(r) = Array::from_shape_vec(out_shape, values) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
