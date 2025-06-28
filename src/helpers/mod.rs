use ndarray::{NdFloat, ScalarOperand};

#[cfg(feature = "ndarray-linalg")]
use ndarray_linalg::{Lapack, Scalar};

use num_traits::{AsPrimitive, FromPrimitive, NumCast, Signed};
use rand::distr::uniform::SampleUniform;

use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

// Include submodules
mod common;
mod distance;

// Re-export types from submodules
pub use common::DataPoint;
pub use distance::{Distance, L1Dist, L2Dist, LInfDist, LpDist};

pub trait Float:
    NdFloat
    + FromPrimitive
    + Default
    + Signed
    + Sum
    + AsPrimitive<usize>
    + for<'a> AddAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + SampleUniform
    + ScalarOperand
    + std::marker::Unpin
{
    #[cfg(feature = "ndarray-linalg")]
    type Lapack: Float + Scalar + Lapack;
    #[cfg(not(feature = "ndarray-linalg"))]
    type Lapack: Float;

    fn cast<T: NumCast>(x: T) -> Option<Self> {
        NumCast::from(x)
    }
}

impl Float for f32 {
    type Lapack = f32;
}

impl Float for f64 {
    type Lapack = f64;
}
