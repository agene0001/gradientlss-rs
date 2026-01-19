//! Core data types for representing response variables.

use ndarray::{ArrayView1, ArrayView2};

/// Enum to represent different structures of response variables.
///
/// This allows the framework to handle both univariate (single-column) and
/// multivariate (multi-column) targets in a type-safe way.
pub enum ResponseData<'a> {
    /// A 1-dimensional view of a univariate target variable.
    Univariate(&'a ArrayView1<'a, f64>),
    /// A 2-dimensional view of a multivariate target variable.
    Multivariate(&'a ArrayView2<'a, f64>),
}
