use crate::trace_fn;
use std::fmt;
use tracing::{debug, error};

/// Represents the shape of a tensor as a list of dimensions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
}

impl Shape {
    /// Creates a new shape from dimensions, computing strides in row-major order
    pub fn new(dims: &[usize]) -> Self {
        trace_fn!("Shape::new");
        debug!("Creating new shape with dims: {:?}", dims);

        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides.get(i + 1).copied().unwrap_or(1) * dims.get(i + 1).copied().unwrap_or(1);
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

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        trace_fn!("Shape::ndim");
        self.dims.len()
    }

    /// Returns the total number of elements
    pub fn len(&self) -> usize {
        trace_fn!("Shape::len");
        self.size
    }

    /// Checks if the shape is empty
    pub fn is_empty(&self) -> bool {
        trace_fn!("Shape::is_empty");
        self.size == 0
    }

    /// Returns the dimensions as a slice
    pub fn dims(&self) -> &[usize] {
        trace_fn!("Shape::dims");
        &self.dims
    }

    /// Returns the strides as a slice
    pub fn strides(&self) -> &[usize] {
        trace_fn!("Shape::strides");
        &self.strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.strides(), &[12, 4, 1]);
        assert_eq!(shape.len(), 24);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    #[should_panic(expected = "Shape must have at least one dimension")]
    fn test_empty_shape() {
        Shape::new(&[]);
    }

    #[test]
    #[should_panic(expected = "Shape dimensions cannot be zero")]
    fn test_zero_dimension() {
        Shape::new(&[1, 0, 3]);
    }
}
