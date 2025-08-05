//! # Tensor Shape
//!
//! Defines the shape and memory layout of tensors, including dimension tracking
//! and stride calculation for efficient memory access patterns.

use crate::trace_fn;
use tracing::{debug, error};

/// Tensor shape and memory layout information.
///
/// Represents both the logical shape (dimensions) and physical memory layout (strides)
/// of a tensor. Used to enable efficient element access and operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
}

impl Shape {
    /// Creates a new shape with automatic stride calculation.
    ///
    /// # Arguments
    /// * `dims` - Slice of dimension sizes
    ///
    /// # Panics
    /// - If `dims` is empty
    /// - If any dimension is zero
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// let shape = Shape::new(&[2, 3]);  // 2x3 matrix
    /// assert_eq!(shape.dims(), &[2, 3]);
    /// ```
    pub fn new(dims: &[usize]) -> Self {
        trace_fn!("Shape::new");
        debug!("Creating new shape with dims: {:?}", dims);

        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] =
                strides.get(i + 1).copied().unwrap_or(1) * dims.get(i + 1).copied().unwrap_or(1);
        }

        let size = dims.iter().product();

        // Validate shape
        if dims.is_empty() {
            error!("Attempted to create shape with empty dimensions");
            panic!("Shape must have at least one dimension");
        }

        if dims.contains(&0) {
            error!("Attempted to create shape with zero dimension: {:?}", dims);
            panic!("Shape dimensions cannot be zero");
        }

        Shape {
            dims: dims.to_vec(),
            strides,
            size,
        }
    }

    /// Returns the rank (number of dimensions) of the tensor.
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// let shape = Shape::new(&[2, 3, 4]);
    /// assert_eq!(shape.ndim(), 3);
    /// ```
    pub fn ndim(&self) -> usize {
        trace_fn!("Shape::ndim");
        self.dims.len()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is the product of all dimension sizes.
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// let shape = Shape::new(&[2, 3, 4]);
    /// assert_eq!(shape.len(), 24);
    /// ```
    pub fn len(&self) -> usize {
        trace_fn!("Shape::len");
        self.size
    }

    /// Returns `true` if the tensor contains no elements.
    ///
    /// A tensor is considered empty if any dimension has size zero.
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// // A shape with no elements is considered empty
    /// let shape = Shape::new(&[1, 2, 3]);
    /// assert!(!shape.is_empty());
    ///
    /// // Note: A shape with any dimension of size 0 would be considered empty,
    /// // but Shape::new() panics when given zero dimensions as it's considered invalid
    /// ```
    pub fn is_empty(&self) -> bool {
        trace_fn!("Shape::is_empty");
        self.size == 0
    }

    /// Returns the shape's dimensions as a slice.
    ///
    /// The dimensions are stored in row-major order (C-style).
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// let shape = Shape::new(&[2, 3]);
    /// assert_eq!(shape.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        trace_fn!("Shape::dims");
        &self.dims
    }

    /// Returns the stride for each dimension in elements.
    ///
    /// The stride is the number of elements to skip in memory to move to the
    /// next element in that dimension.
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::Shape;
    /// let shape = Shape::new(&[2, 3]);
    /// assert_eq!(shape.strides(), &[3, 1]);  // For row-major order
    /// ```
    pub fn strides(&self) -> &[usize] {
        trace_fn!("Shape::strides");
        &self.strides
    }
}
