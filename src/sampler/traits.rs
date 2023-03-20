use super::common::SamplingMode;
use nalgebra::{clamp, MatrixXx3};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};

/// This trait has to be implented by all valid samplers.
///
pub trait ReSample<T, U>
where
    T: Num + AsPrimitive<usize> + AsPrimitive<U> + PartialOrd + Copy,
    U: Num + Copy + 'static,
    usize: AsPrimitive<T>,
{
    fn set_sampling_mode(&mut self, mode: SamplingMode);
    fn get_sampling_mode(&self) -> SamplingMode;

    fn set_cval(&mut self, cval: U);
    fn get_cval(&self) -> U;

    fn sample(
        &self,
        in_im: &Array<U, IxDyn>,
        in_coords: &mut MatrixXx3<T>,
        out_shape: &[usize],
    ) -> Result<Array<U, IxDyn>, String>;

    fn apply_sampling_mode(&self, in_im: &Array<U, IxDyn>, in_coords: &mut MatrixXx3<T>) {
        let in_shape = in_im.shape();

        let caps: [T; 3] = [
            (in_shape[0] - 1).as_(),
            (in_shape[1] - 1).as_(),
            (in_shape[2] - 1).as_(),
        ];

        match self.get_sampling_mode() {
            SamplingMode::Constant => (), // leave idxs as is
            SamplingMode::Nearest => {
                for (i, mut col) in in_coords.column_iter_mut().enumerate() {
                    col.iter_mut()
                        .for_each(|x| x.clone_from(&clamp(*x, T::zero(), caps[i])))
                }
            }
        }
    }

    fn get_val(&self, im: &Array<U, IxDyn>, x: T, y: T, z: T) -> U
    {
        for ax in [x, y, z] {
            if ax < T::zero() {
                return self.get_cval();
            }
        }

        match im.get([x.as_(), y.as_(), z.as_()]) {
            Some(val) => *val,
            None => self.get_cval(),
        }
    }
}
