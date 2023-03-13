use super::common::SamplingMode;
use super::traits::ReSample;
use nalgebra::{clamp, MatrixXx3};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};

/// A sampler employing a nearest neighbor strategy.
///
/// This sampler corresponds to `order=0` in nibabel.
///
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
    T: Num + AsPrimitive<usize> + PartialOrd + Copy,
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
        in_coords: &MatrixXx3<T>,
        out_shape: &[usize],
    ) -> Result<Array<U, IxDyn>, String> {
        let mut values: Vec<U> = Vec::with_capacity(in_coords.len());

        let in_shape = in_im.shape();

        let (cap_x, cap_y, cap_z) = (
            (in_shape[0] - 1).as_(),
            (in_shape[1] - 1).as_(),
            (in_shape[2] - 1).as_(),
        );

        'outer: for in_coord in in_coords.row_iter() {
            let (mut x, mut y, mut z) = (in_coord[(0, 0)], in_coord[(0, 1)], in_coord[(0, 2)]);

            // handle different out of sample modes
            #[allow(unreachable_patterns)]
            match self.mode {
                SamplingMode::Constant => (), // leave idxs as is
                SamplingMode::Nearest => {
                    x = clamp(x, T::zero(), cap_x);
                    y = clamp(y, T::zero(), cap_y);
                    z = clamp(z, T::zero(), cap_z);
                }
                _ => unimplemented!("Mode: {:?} is not implemented!", self.mode),
            }

            for ax in [x, y, z] {
                if ax < T::zero() {
                    values.push(U::zero()); // ToDo cval
                    continue 'outer;
                }
            }

            let val = match in_im.get([x.as_(), y.as_(), z.as_()]) {
                Some(val) => *val,
                None => U::zero(), // ToDo cval
            };
            values.push(val);
        }

        if let Ok(r) = Array::from_shape_vec(
            [
                out_shape[0] as usize,
                out_shape[1] as usize,
                out_shape[2] as usize,
            ],
            values,
        ) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
