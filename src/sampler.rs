use crate::Matrix3N;
use nalgebra as na;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};

#[derive(Debug)]
pub enum Mode {
    Constant,
    Nearest,
}

pub trait Sampler<T, U>
where
    T: Num + Copy,
    U: Num + Copy,
{
    fn sample(
        &self,
        in_im: &Array<U, IxDyn>,
        in_coords: &Matrix3N<T>,
        out_shape: &[u16],
    ) -> Array<U, IxDyn>;
}

pub struct NearestNeighbor<U>
where
    U: Num + Copy,
{
    mode: Mode,
    cval: U,
}

impl<U> Default for NearestNeighbor<U>
where
    U: Num + Copy,
{
    fn default() -> Self {
        Self {
            mode: Mode::Constant,
            cval: U::zero(),
        }
    }
}

impl<T, U> Sampler<T, U> for NearestNeighbor<U>
where
    T: Num + AsPrimitive<i32> + Copy,
    U: Num + Copy,
{
    fn sample(
        &self,
        in_im: &Array<U, IxDyn>,
        in_coords: &Matrix3N<T>,
        out_shape: &[u16],
    ) -> Array<U, IxDyn> {
        let in_coords: Vec<i32> = in_coords.iter().map(|x| x.as_()).collect();
        let mut v: Vec<U> = Vec::with_capacity(in_coords.len());
        let in_coords: Matrix3N<i32> = Matrix3N::from_vec(in_coords);

        let in_shape = in_im.shape();

        //println!("nn: \n{}", in_coords.rows(0, 10));
        //println!("out_shape: {:?}", out_shape);
        //println!("in_shape: {:?}", in_shape);

        let (cap_x, cap_y, cap_z) = (
            (in_shape[0] - 1) as i32,
            (in_shape[1] - 1) as i32,
            (in_shape[2] - 1) as i32,
        );

        'outer: for in_coord in in_coords.row_iter() {
            let (mut x, mut y, mut z) = (in_coord[(0, 0)], in_coord[(0, 1)], in_coord[(0, 2)]);

            // handle different out of sample modes
            #[allow(unreachable_patterns)]
            match self.mode {
                Mode::Constant => (), // leave idxs as is
                Mode::Nearest => {
                    x = na::clamp(x, 0, cap_x);
                    y = na::clamp(y, 0, cap_y);
                    z = na::clamp(z, 0, cap_z);
                }
                _ => unimplemented!("Mode: {:?} is not implemented!", self.mode),
            }

            for ax in [x, y, z] {
                if ax < 0 {
                    v.push(U::zero()); // ToDo cval
                    continue 'outer;
                }
            }

            let val = match in_im.get([x as usize, y as usize, z as usize]) {
                Some(val) => *val,
                None => U::zero(), // ToDo cval
            };
            v.push(val);
        }

        Array::from_shape_vec(
            [
                out_shape[0] as usize,
                out_shape[1] as usize,
                out_shape[2] as usize,
            ],
            v,
        )
        .unwrap()
        .into_dyn()
    }
}
