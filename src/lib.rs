//! This library is an extension of the NIFTI-rs library, adding resampling support.
//! This library is closely modeled after the NiBabel processing module, hence the name.

use itertools::Itertools;
use nalgebra::{ClosedAdd, ClosedMul, Matrix3, Matrix4, MatrixXx3, RealField, Scalar, Vector3};
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num};
use std::fmt::Display;

pub mod sampler;
pub use sampler::nearest_neighbor::NearestNeighbor;
pub use sampler::traits::ReSample;

/// Corners could be calculated, e.g. using itertools. As we only need to cover the 3D
/// usecase, we simply do it "manually".
#[rustfmt::skip] // do not mangle manual matrix format
fn get_corners(input_shape: &Vector3<usize>) -> MatrixXx3<usize>
where
{
    let is = input_shape;
    MatrixXx3::<usize>::from_row_slice(&[
        0,         0,        0,
        0,         0,        is[2] - 1,
        0,         is[1] -1, 0,
        0,         is[1] -1, is[2] - 1,
        is[0] - 1, 0,        0,
        is[0] - 1, 0,        is[2] - 1,
        is[0] - 1, is[1] -1, 0,
        is[0] - 1, is[1] -1, is[2] - 1,
    ])
}

/// Transform the point matrix by applying the affine transform + translation.
fn apply_affine<T>(affine: &Matrix4<T>, pts: &MatrixXx3<T>) -> MatrixXx3<T>
where
    T: Num + Scalar + ClosedAdd + ClosedMul + Copy,
{
    let (aff, tra) = afftra_to_aff_tra(affine);
    let mut r = pts * aff.transpose();
    let _tra = tra.transpose();

    for mut row in r.row_iter_mut() {
        row += &_tra;
    }
    r
}

/// output-aligned shape, affine for input implied by `in_shape` & `in_affine`
///
/// The input (voxel) space, and the affine mapping to output space, are given
/// in `in_shape` & `in_affine`.
///
/// The output space is implied by the affine, we don't need to know what that
/// is, we just return something with the same (implied) output space.
///
/// Our job is to work out another voxel space where the voxel array axes and
/// the output axes are aligned (top left 3 x 3 of affine is diagonal with all
/// positive entries) and which contains all the voxels of the implied input
/// image at their correct output space positions, once resampled into the
/// output voxel space.
///
/// (from the nibabel documentation)
///
fn vox2out_vox<T>(
    in_shape: &Vector3<usize>,
    in_affine: &Matrix4<T>,
    voxel_sizes: &Vector3<T>,
) -> (Vector3<usize>, Matrix4<T>)
where
    T: Num + Scalar + RealField + AsPrimitive<usize> + Copy + Display,
    usize: AsPrimitive<T>,
{
    let in_corners: MatrixXx3<usize> = get_corners(in_shape);
    // convert MatrixXx3<usize> -> MatrixXx3<T>
    let in_corners: MatrixXx3<T> =
        MatrixXx3::from_iterator(in_corners.nrows(), in_corners.iter().map(|x| x.as_()));

    let out_corners: MatrixXx3<T> = apply_affine(in_affine, &in_corners);

    // ToDo make pretty
    let out_mn = out_corners
        .column_iter()
        .map(|x| x.min())
        .collect::<Vec<T>>();
    let out_mn = Vector3::from_vec(out_mn);
    let out_mx = out_corners
        .column_iter()
        .map(|x| x.max())
        .collect::<Vec<T>>();
    let out_mx = Vector3::from_vec(out_mx);

    let out_shape: Vector3<T> = (out_mx - out_mn)
        .component_div(voxel_sizes)
        .map(|x| x.ceil())
        .add_scalar(T::one());
    let out_shape: Vector3<usize> = Vector3::from_iterator(out_shape.iter().map(|x| x.as_()));

    let out_aff = Matrix3::from_diagonal(voxel_sizes);
    let out_tra = out_mn;
    let out_affine = aff_tra_to_afftra(&out_aff, &out_tra);

    (out_shape, out_affine)
}

fn aff_tra_to_afftra<T>(aff: &Matrix3<T>, tra: &Vector3<T>) -> Matrix4<T>
where
    T: Num + Scalar + Copy,
{
    let r = aff.clone();
    let r = r.insert_column(3, T::zero());
    let mut r = r.insert_row(3, T::zero());
    r[(0, 3)] = tra.x;
    r[(1, 3)] = tra.y;
    r[(2, 3)] = tra.z;
    r[(3, 3)] = T::one();
    r
}

fn afftra_to_aff_tra<T>(affine: &Matrix4<T>) -> (Matrix3<T>, Vector3<T>)
where
    T: Num + Scalar + Copy,
{
    let aff = affine.fixed_slice::<3, 3>(0, 0);
    let tra = affine.fixed_slice::<3, 1>(0, 3);
    (aff.into(), tra.into())
}

/// Resample in_im to world space with a given voxel size.
///
pub fn resample_to_output<T, U, S>(
    in_im: &Array<U, IxDyn>,
    in_affine: &Matrix4<T>,
    voxel_sizes: &[f32; 3],
    sampler: S,
) -> Result<(Array<U, IxDyn>, Matrix4<T>), String>
where
    T: Scalar + RealField + nifti::DataElement + AsPrimitive<usize> + Copy,
    U: Num + Copy,
    S: ReSample<T, U> + 'static,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    // ToDo make pretty
    let in_shape = in_im.shape();
    let in_shape = Vector3::from_row_slice(&[in_shape[0], in_shape[1], in_shape[2]]);

    let voxel_sizes: Vector3<T> = Vector3::from_row_slice(&voxel_sizes.map(|x| x.as_()));

    let (out_shape, out_affine) = vox2out_vox(&in_shape, in_affine, &voxel_sizes);
    let out_shape: [usize; 3] = out_shape.into();
    match resample_from_to(in_im, in_affine, &out_shape, &out_affine, sampler) {
        Ok(out_im) => Ok((out_im, out_affine)),
        Err(err) => Err(err),
    }
}

/// Resample in_im to mapped voxel space defined by out_affine and out_shape.
///
pub fn resample_from_to<T, U, S>(
    in_im: &Array<U, IxDyn>,
    in_affine: &Matrix4<T>,
    out_shape: &[usize; 3],
    out_affine: &Matrix4<T>,
    sampler: S,
) -> Result<Array<U, IxDyn>, String>
where
    T: Num + Scalar + RealField + nifti::DataElement + AsPrimitive<usize> + Copy,
    U: Num + Copy,
    S: ReSample<T, U> + 'static,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let inv_in_affine = match in_affine.pseudo_inverse((0.00001f32).as_()) {
        Ok(val) => val,
        Err(err) => return Err(err.into()),
    };

    let compound_affine = inv_in_affine * out_affine;

    // ToDo: generation of all coords is not very fast
    let in_coord_iter = out_shape.iter().map(|x| 0..*x).multi_cartesian_product();
    let in_coords: Vec<usize> = in_coord_iter.flatten().collect_vec();

    // the iterator yields row-major order, nalgebra uses column-major order
    // ToDo: this is slow. Possibly replace with Matrix3N::from_vec,
    //       however, this operation expects column-major order
    let in_coords = MatrixXx3::from_row_slice(&in_coords);
    // convert MatrixXx3<usize> -> MatrixXx3<T>
    let in_coords: MatrixXx3<T> =
        MatrixXx3::from_iterator(in_coords.nrows(), in_coords.iter().map(|x| x.as_()));

    let out_coords = apply_affine(&compound_affine, &in_coords);

    match sampler.sample(in_im, &out_coords, out_shape) {
        Ok(out_im) => Ok(out_im),
        Err(err) => Err(err),
    }
}
