use super::common::SamplingMode;
use super::traits::ReSample;
use nalgebra::{MatrixXx3, RealField};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};
use rayon::prelude::*;

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
    T: Num + AsPrimitive<usize> + AsPrimitive<U> + RealField + PartialOrd + Copy,
    U: Num + AsPrimitive<T> + Copy + Send + Sync,
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

    #[allow(non_snake_case)]
    fn sample(
        &self,
        in_im: &Array<U, IxDyn>,
        in_coords: &mut MatrixXx3<T>,
        out_shape: &[usize],
    ) -> Result<Array<U, IxDyn>, String> {
        self.apply_sampling_mode(in_im, in_coords);

        let in_shape = in_im.shape();
        let t_zero   = T::zero();
        let t_one    = T::one();
        let x_upper  = T::from_usize(in_shape[0]).expect("failed to determine upper X");
        let y_upper  = T::from_usize(in_shape[1]).expect("failed to determine upper Y");
        let z_upper  = T::from_usize(in_shape[2]).expect("failed to determine upper Z");

        let in_coords_0 = MatrixXx3::from_vec(in_coords.as_slice().into_par_iter().map(|x| x.floor()).collect());
        let in_coords_1 = MatrixXx3::from_vec(in_coords_0.as_slice().into_par_iter().map(|x| *x + t_one).collect());

        let values: Vec<U> = (0..in_coords.nrows()).into_par_iter().map(|i| {
            let (x, y, z) = (in_coords[(i, 0)], in_coords[(i, 1)], in_coords[(i, 2)]);

            // check if index is out of bounds
            if  // check if any of the coordinates are out of lower bounds
                (x < t_zero)  | (y < t_zero)  | (z < t_zero) |
                // check if any of the coordinates are out of upper bounds
                (x > x_upper) | (y > y_upper) | (z > z_upper)
            {
                return self.get_cval();
            };

            let (x0, y0, z0) = (
                in_coords_0[(i, 0)],
                in_coords_0[(i, 1)],
                in_coords_0[(i, 2)],
            );
            let (x1, y1, z1) = (
                in_coords_1[(i, 0)],
                in_coords_1[(i, 1)],
                in_coords_1[(i, 2)],
            );

            // simd does not play nice with num_traits
            // we convert to usize here instead of in
            // advance to leaverage multiprocessing.
            // if implemented with simd there would be
            // a total expected speedup of 5 % for
            // overall resampling
            let (x0_u, y0_u, z0_u) = (
                in_coords_0[(i, 0)].as_(),
                in_coords_0[(i, 1)].as_(),
                in_coords_0[(i, 2)].as_(),
            );
            let (x1_u, y1_u, z1_u) = (
                in_coords_1[(i, 0)].as_(),
                in_coords_1[(i, 1)].as_(),
                in_coords_1[(i, 2)].as_(),
            );

            let Ia = self.get_val(in_im, x0_u, y0_u, z0_u);
            let Ib = self.get_val(in_im, x0_u, y0_u, z1_u);
            let Ic = self.get_val(in_im, x0_u, y1_u, z0_u);
            let Id = self.get_val(in_im, x0_u, y1_u, z1_u);
            let Ie = self.get_val(in_im, x1_u, y0_u, z0_u);
            let If = self.get_val(in_im, x1_u, y0_u, z1_u);
            let Ig = self.get_val(in_im, x1_u, y1_u, z0_u);
            let Ih = self.get_val(in_im, x1_u, y1_u, z1_u);

            let wa: U = ((x1 - x) * (y1 - y) * (z1 - z)).as_();
            let wb: U = ((x1 - x) * (y1 - y) * (z - z0)).as_();
            let wc: U = ((x1 - x) * (y - y0) * (z1 - z)).as_();
            let wd: U = ((x1 - x) * (y - y0) * (z - z0)).as_();
            let we: U = ((x - x0) * (y1 - y) * (z1 - z)).as_();
            let wf: U = ((x - x0) * (y1 - y) * (z - z0)).as_();
            let wg: U = ((x - x0) * (y - y0) * (z1 - z)).as_();
            let wh: U = ((x - x0) * (y - y0) * (z - z0)).as_();

            wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
        })
        .collect();

        if let Ok(r) = Array::from_shape_vec(out_shape, values) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
