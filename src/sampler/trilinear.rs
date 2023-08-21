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
    T: Num + AsPrimitive<usize> + AsPrimitive<U> + RealField + PartialOrd + Copy,
    U: Num + AsPrimitive<T> + Copy,
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
        let mut values: Vec<U> = Vec::with_capacity(in_coords.len());
        self.apply_sampling_mode(in_im, in_coords);

        let in_coords_0 =
            MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter().map(|x| x.floor()));
        let in_coords_1 = MatrixXx3::from_iterator(
            in_coords_0.nrows(),
            in_coords_0.iter().map(|x| *x + T::one()),
        );

        for i in 0..in_coords.nrows() {
            let (x, y, z) = (in_coords[(i, 0)], in_coords[(i, 1)], in_coords[(i, 2)]);
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

            let Ia = self.get_val(in_im, x0, y0, z0);
            let Ib = self.get_val(in_im, x0, y0, z1);
            let Ic = self.get_val(in_im, x0, y1, z0);
            let Id = self.get_val(in_im, x0, y1, z1);
            let Ie = self.get_val(in_im, x1, y0, z0);
            let If = self.get_val(in_im, x1, y0, z1);
            let Ig = self.get_val(in_im, x1, y1, z0);
            let Ih = self.get_val(in_im, x1, y1, z1);

            let wa: U = ((x1 - x) * (y1 - y) * (z1 - z)).as_();
            let wb: U = ((x1 - x) * (y1 - y) * (z - z0)).as_();
            let wc: U = ((x1 - x) * (y - y0) * (z1 - z)).as_();
            let wd: U = ((x1 - x) * (y - y0) * (z - z0)).as_();
            let we: U = ((x - x0) * (y1 - y) * (z1 - z)).as_();
            let wf: U = ((x - x0) * (y1 - y) * (z - z0)).as_();
            let wg: U = ((x - x0) * (y - y0) * (z1 - z)).as_();
            let wh: U = ((x - x0) * (y - y0) * (z - z0)).as_();

            values.push(
                wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih,
            );
        }

        if let Ok(r) = Array::from_shape_vec(out_shape, values) {
            Ok(r.into_dyn())
        } else {
            Err("number of elements is not compatible with out_shape shape".into())
        }
    }
}
