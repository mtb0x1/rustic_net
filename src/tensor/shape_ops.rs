//! Shape manipulation operations for Tensor
//!
//! This module contains methods that manipulate the shape of tensors without changing their data.

use super::{Device, DType, Shape, Tensor};
use crate::trace_fn;
use std::sync::Arc;

impl Tensor {
    /// Reshapes the tensor to the given shape
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, String> {
        trace_fn!("tensor::shape_ops::reshape");
        
        // Calculate the total number of elements in the current and new shapes
        let current_size: usize = self.shape.dims().iter().product();
        let new_size: usize = new_shape.iter().product();
        
        // Validate that the total number of elements matches
        if current_size != new_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?}",
                current_size, new_shape
            ));
        }
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            dtype: self.dtype,
        })
    }
    
    /// Transposes the tensor by swapping the last two dimensions
    pub fn t(&self) -> Tensor {
        trace_fn!("tensor::shape_ops::t");
        self.transpose()
    }
    
    /// Transposes the tensor by swapping the last two dimensions
    pub fn transpose(&self) -> Tensor {
        trace_fn!("tensor::shape_ops::transpose");
        let dims = self.shape.dims();
        let rank = dims.len();
        
        if rank < 2 {
            // For 0D or 1D tensors, transpose is a no-op
            return self.clone();
        }
        
        // Create a new shape with the last two dimensions swapped
        let mut new_dims = dims.to_vec();
        new_dims.swap(rank - 2, rank - 1);
        
        // For the data, we need to reorder the elements
        let total_elements: usize = dims.iter().product();
        let m = dims[rank - 2];
        let n = dims[rank - 1];
        
        // For higher dimensions, we need to iterate over all but the last two dimensions
        let higher_dims: usize = if rank > 2 { dims[..rank - 2].iter().product() } else { 1 };
        
        let mut new_data = Vec::with_capacity(total_elements);
        
        for h in 0..higher_dims {
            for i in 0..m {
                for j in 0..n {
                    let idx = h * m * n + i * n + j;
                    new_data.push(self.data[idx]);
                }
            }
        }
        
        Tensor {
            data: Arc::new(new_data),
            shape: Shape::new(&new_dims),
            device: self.device,
            dtype: self.dtype,
        }
    }
    
    /// Adds a new dimension of size 1 at the specified axis
    pub fn expand_dims(&self, axis: usize) -> Result<Tensor, String> {
        trace_fn!("tensor::shape_ops::expand_dims");
        let rank = self.rank();
        
        if axis > rank {
            return Err(format!(
                "Axis {} is out of bounds for tensor of rank {}",
                axis, rank
            ));
        }
        
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(axis, 1);
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(&new_dims),
            device: self.device,
            dtype: self.dtype,
        })
    }
    
    /// Removes dimensions of size 1
    pub fn squeeze(&self, axis: Option<usize>) -> Tensor {
        trace_fn!("tensor::shape_ops::squeeze");
        let dims = self.shape.dims();
        let mut new_dims = Vec::new();
        
        match axis {
            Some(axis) => {
                if axis >= dims.len() {
                    panic!("Axis {} is out of bounds for tensor of rank {}", axis, dims.len());
                }
                
                for (i, &dim) in dims.iter().enumerate() {
                    if i != axis || dim != 1 {
                        new_dims.push(dim);
                    }
                }
            }
            None => {
                // Remove all dimensions of size 1
                for &dim in dims {
                    if dim != 1 {
                        new_dims.push(dim);
                    }
                }
            }
        }
        
        // If we removed all dimensions, ensure we have at least one dimension
        if new_dims.is_empty() {
            new_dims.push(1);
        }
        
        Tensor {
            data: self.data.clone(),
            shape: Shape::new(&new_dims),
            device: self.device,
            dtype: self.dtype,
        }
    }
    
    /// Returns the number of dimensions of the tensor
    pub fn rank(&self) -> usize {
        self.shape.dims().len()
    }
    
    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }
    
    /// Returns the total number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.dims().iter().product()
    }
}
