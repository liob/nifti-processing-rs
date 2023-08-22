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

        self.apply_sampling_mode(in_im, in_coords);
        let in_coords =
            MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter_mut().map(|x| x.ceil()));
        let in_coords_u: MatrixXx3<usize> = MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter().map(|x| x.as_()));

        let in_shape = in_im.shape();
        let t_zero  = T::zero();
        let x_upper = T::from_usize(in_shape[0]).expect("failed to determine upper X");
        let y_upper = T::from_usize(in_shape[1]).expect("failed to determine upper Y");
        let z_upper = T::from_usize(in_shape[2]).expect("failed to determine upper Z");

        for i in 0..in_coords.nrows() {
            let (x, y, z) = (in_coords[(i, 0)], in_coords[(i, 1)], in_coords[(i, 2)]);
            let (x_u, y_u, z_u) = (in_coords_u[(i, 0)], in_coords_u[(i, 1)], in_coords_u[(i, 2)]);

            // check if index is out of bounds
            if  // check if any of the coordinates are out of lower bounds
                (x < t_zero)  | (y < t_zero)  | (z < t_zero) |
                // check if any of the coordinates are out of upper bounds
                (x > x_upper) | (y > y_upper) | (z > z_upper)
            {
                values.push(self.get_cval());
                continue;
            };

            values.push(self.get_val(in_im, x_u, y_u, z_u));
        }

        if let Ok(r) = Array::from_shape_vec(out_shape, values) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
