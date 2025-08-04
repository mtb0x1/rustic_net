use crate::trace_fn;
use rand::Rng;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use tracing::{debug, error};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Represents the device where tensor data is stored
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Device {
    /// CPU device with optional device ID (useful for multi-CPU systems)
    Cpu(Option<usize>),
    /// CUDA device with device ID
    Cuda(usize),
    /// WebGPU device with device ID
    WebGpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        trace_fn!("Device::default");
        debug!("Creating default CPU device with ID 0");
        // Default to CPU device with ID 0
        Device::Cpu(Some(0))
    }
}

/// Represents the data type of tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit floating point
    #[default]
    F32,
}

impl DType {
    /// Returns the size in bytes of the data type
    pub fn size_of(&self) -> usize {
        trace_fn!("DType::size_of");
        debug!("Getting size of DType: {:?}", self);
        match self {
            DType::F32 => 4,
        }
    }
}

impl TryFrom<&str> for DType {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        trace_fn!("DType::try_from");
        debug!("Attempting to convert string '{}' to DType", value);
        match value {
            "f32" => Ok(DType::F32),
            _ => Err(format!("Unsupported data type: {value}")),
        }
    }
}

impl TryFrom<DType> for &str {
    type Error = String;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::F32 => Ok("f32"),
        }
    }
}

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
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        let size = dims.iter().product();

        // Validate shape
        if dims.is_empty() {
            error!("Attempted to create shape with empty dimensions");
            panic!("Shape must have at least one dimension");
        }

        if dims.contains(&0) {
            error!("Attempted to create shape with zero dimension: {:?}", dims);
            panic!("All dimensions must be greater than 0");
        }

        let shape = Shape {
            dims: dims.to_vec(),
            strides,
            size,
        };

        debug!("Created shape: {:?}", shape);
        shape
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        trace_fn!("Shape::ndim");
        debug!("Getting number of dimensions for shape: {:?}", self.dims);
        self.dims.len()
    }

    /// Returns the total number of elements
    pub fn len(&self) -> usize {
        trace_fn!("Shape::len");
        debug!(
            "Getting total number of elements for shape: {:?}",
            self.dims
        );
        self.size
    }

    /// Checks if the shape is empty
    pub fn is_empty(&self) -> bool {
        trace_fn!("Shape::is_empty");
        let empty = self.size == 0;
        debug!("Shape {:?} is empty: {}", self.dims, empty);
        empty
    }

    /// Returns the dimensions as a slice
    pub fn dims(&self) -> &[usize] {
        trace_fn!("Shape::dims");
        debug!("Getting dimensions for shape: {:?}", self.dims);
        &self.dims
    }

    /// Returns the strides as a slice
    pub fn strides(&self) -> &[usize] {
        trace_fn!("Shape::strides");
        debug!(
            "Getting strides for shape: {:?} -> {:?}",
            self.dims, self.strides
        );
        &self.strides
    }
}

/// Represents a multi-dimensional array (tensor)
pub struct Tensor {
    /// The underlying data buffer
    data: Vec<f32>,
    /// The shape of the tensor
    shape: Shape,
    /// The device where the tensor is stored
    device: Device,
    /// The data type of tensor elements
    dtype: DType,
}

impl Tensor {
    /// Creates a new tensor from a vector with the given shape and device
    pub fn from_vec<T: Into<Vec<f32>>>(
        data: T,
        shape: &[usize],
        device: Device,
    ) -> Result<Self, String> {
        trace_fn!("Tensor::from_vec");
        debug!(
            "Creating tensor with shape: {:?}, device: {:?}",
            shape, device
        );

        let data = data.into();
        let expected_len: usize = shape.iter().product();

        if data.len() != expected_len {
            let err = format!(
                "Data length {} does not match shape {:?} (expected {} elements)",
                data.len(),
                shape,
                expected_len
            );
            error!("{}", err);
            return Err(err);
        }

        let tensor = Tensor {
            data,
            shape: Shape::new(shape),
            device,
            dtype: DType::F32,
        };

        debug!(
            "Created tensor with shape: {:?}, device: {:?}",
            tensor.shape(),
            tensor.device()
        );
        Ok(tensor)
    }

    /// Creates a new tensor filled with ones
    pub fn ones(shape: &[usize], device: Device) -> Self {
        trace_fn!("Tensor::ones");
        debug!(
            "Creating ones tensor with shape: {:?}, device: {:?}",
            shape, device
        );

        let len: usize = shape.iter().product();
        let tensor = Tensor {
            data: vec![1.0; len],
            shape: Shape::new(shape),
            device,
            dtype: DType::F32,
        };

        debug!("Created ones tensor with shape: {:?}", tensor.shape());
        tensor
    }

    /// Creates an identity matrix of the given size
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, the matrix initialization
    /// will be parallelized for better performance with large matrices.
    pub fn identity(size: usize, device: Device) -> Self {
        trace_fn!("Tensor::identity");
        debug!("Creating identity matrix of size: {}", size);

        #[cfg(feature = "parallel")]
        {
            debug!("Using parallel identity matrix initialization");

            let data: Vec<f32> = (0..size * size)
                .into_par_iter()
                .map(|i| {
                    //let curr_thread = std::thread::current().id();
                    //debug!("identity ====> Thread {:?} processing element {}", curr_thread, i);
                    let row = i / size;
                    let col = i % size;
                    if row == col {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let tensor = Tensor {
                data,
                shape: Shape::new(&[size, size]),
                device,
                dtype: DType::F32,
            };

            debug!(
                "Created parallel identity matrix with shape: {:?}",
                tensor.shape()
            );
            tensor
        }

        #[cfg(not(feature = "parallel"))]
        {
            debug!("Using sequential identity matrix initialization");
            let mut data = vec![0.0; size * size];
            for i in 0..size {
                data[i * size + i] = 1.0;
            }

            let tensor = Tensor {
                data,
                shape: Shape::new(&[size, size]),
                device,
                dtype: DType::F32,
            };

            debug!(
                "Created sequential identity matrix with shape: {:?}",
                tensor.shape()
            );
            tensor
        }
    }

    /// Creates a new tensor with random values between 0.0 and 1.0
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, random number generation
    /// will be parallelized for better performance with large tensors.
    pub fn random(shape: &[usize], device: Device) -> Self {
        trace_fn!("Tensor::random");
        debug!("Creating random tensor with shape: {:?}", shape);

        let len: usize = shape.iter().product();

        #[cfg(feature = "parallel")]
        {
            debug!("Using parallel random number generation");
            use rand::distr::Uniform;
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            let uniform = match Uniform::new(0.0f32, 1.0) {
                Ok(uniform) => uniform,
                Err(e) => {
                    error!("Failed to create uniform distribution: {}", e);
                    panic!("Failed to create uniform distribution: {e}");
                }
            };

            // Generate random numbers in parallel with independent RNGs per thread
            let data: Vec<f32> = (0..len)
                .into_par_iter()
                .map_init(
                    || StdRng::from_rng(&mut rand::rng()), // Per-thread RNG
                    |rng, _| rng.sample(uniform),
                )
                .collect();

            let tensor = Tensor {
                data,
                shape: Shape::new(shape),
                device,
                dtype: DType::F32,
            };

            debug!(
                "Created parallel random tensor with shape: {:?}",
                tensor.shape()
            );
            tensor
        }

        #[cfg(not(feature = "parallel"))]
        {
            debug!("Using sequential random number generation");
            let mut rng = rand::rng();
            let data: Vec<f32> = (0..len).map(|_| rng.random_range(0.0..1.0)).collect();

            let tensor = Tensor {
                data,
                shape: Shape::new(shape),
                device,
                dtype: DType::F32,
            };

            debug!(
                "Created sequential random tensor with shape: {:?}",
                tensor.shape()
            );
            tensor
        }
    }

    /// Creates a new 1D tensor with values from start to end (exclusive) with step size 1
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, the range generation
    /// will be parallelized for better performance with large ranges.
    pub fn arange(start: f32, end: f32) -> Self {
        trace_fn!("Tensor::arange");
        debug!("Creating arange tensor from {} to {}", start, end);

        let len = (end - start) as usize;

        #[cfg(feature = "parallel")]
        {
            debug!("Using parallel arange generation");
            let chunk_size = (len / rayon::current_num_threads().max(1)).max(1);
            let data: Vec<f32> = (0..len)
                .into_par_iter()
                .with_min_len(chunk_size)
                .map(|i| start + i as f32)
                .collect();

            let tensor = Tensor {
                data,
                shape: Shape::new(&[len]),
                device: Device::default(),
                dtype: DType::F32,
            };

            debug!(
                "Created parallel arange tensor with shape: {:?}",
                tensor.shape()
            );
            tensor
        }

        #[cfg(not(feature = "parallel"))]
        {
            debug!("Using sequential arange generation");
            let data: Vec<f32> = (0..len).map(|i| start + i as f32).collect();

            let tensor = Tensor {
                data,
                shape: Shape::new(&[len]),
                device: Device::default(),
                dtype: DType::F32,
            };

            debug!(
                "Created sequential arange tensor with shape: {:?}",
                tensor.shape()
            );
            tensor
        }
    }

    /// Creates a new tensor from a slice with the given shape
    pub fn from_slice(slice: &[f32], shape: &[usize], device: Device) -> Result<Self, String> {
        trace_fn!("Tensor::from_slice");
        debug!("Creating tensor from slice with shape: {:?}", shape);

        let expected_len: usize = shape.iter().product();
        if slice.len() != expected_len {
            let err = format!(
                "Slice length {} does not match shape {:?} (expected {} elements)",
                slice.len(),
                shape,
                expected_len
            );
            error!("{}", err);
            return Err(err);
        }

        let tensor = Tensor {
            data: slice.to_vec(),
            shape: Shape::new(shape),
            device,
            dtype: DType::F32,
        };

        debug!("Created tensor from slice with shape: {:?}", tensor.shape());
        Ok(tensor)
    }

    /// Creates a new tensor filled with zeros
    pub fn zeros(shape: &[usize], device: Device) -> Self {
        trace_fn!("Tensor::zeros");
        debug!(
            "Creating zeros tensor with shape: {:?}, device: {:?}",
            shape, device
        );

        let len: usize = shape.iter().product();
        let tensor = Tensor {
            data: vec![0.0; len],
            shape: Shape::new(shape),
            device,
            dtype: DType::F32,
        };

        debug!("Created zeros tensor with shape: {:?}", tensor.shape());
        tensor
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        trace_fn!("Tensor::shape");
        self.shape.dims()
    }

    /// Returns the device where the tensor is stored
    pub fn device(&self) -> &Device {
        trace_fn!("Tensor::device");
        &self.device
    }

    /// Returns the data type of tensor elements
    pub fn dtype(&self) -> DType {
        trace_fn!("Tensor::dtype");
        self.dtype
    }

    /// Reshapes the tensor to the given shape
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        trace_fn!("Tensor::reshape");
        debug!(
            "Reshaping tensor from {:?} to {:?}",
            self.shape(),
            new_shape
        );

        let current_size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            let err = format!(
                "Cannot reshape tensor of size {current_size} to shape {new_shape:?} (size {new_size})",
            );
            error!("{}", err);
            return Err(err);
        }

        let mut result = self.clone();
        result.shape = Shape::new(new_shape);
        Ok(result)
    }

    /// Transposes the tensor according to the given axes
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        trace_fn!("Tensor::transpose");

        let ndim = self.shape().len();
        let axes = axes.map_or_else(|| (0..ndim).rev().collect::<Vec<_>>(), |ax| ax.to_vec());

        if axes.len() != ndim {
            let err = format!("Axes {axes:?} don't match tensor dimensions {ndim}");
            error!("{}", err);
            return Err(err);
        }

        // For 2D tensors, we can optimize the transpose
        if ndim == 2 {
            let rows = self.shape()[0];
            let cols = self.shape()[1];
            let mut data = vec![0.0; self.data.len()];

            #[cfg(feature = "parallel")]
            {
                debug!("Using parallel 2D transpose ({}x{})", rows, cols);
                let src = &self.data;
                data.par_chunks_mut(rows).enumerate().for_each(|(j, col)| {
                    for i in 0..rows {
                        col[i] = src[i * cols + j];
                    }
                });
            }

            #[cfg(not(feature = "parallel"))]
            {
                debug!("Using sequential 2D transpose ({}x{})", rows, cols);
                for i in 0..rows {
                    for j in 0..cols {
                        data[j * rows + i] = self.data[i * cols + j];
                    }
                }
            }

            return Tensor::from_vec(data, &[cols, rows], self.device);
        }

        // For higher dimensions, we need a more general approach
        // This is a placeholder - full ND transpose is more complex
        // and would require a more sophisticated implementation
        debug!("Transposing tensor with axes: {:?}", axes);
        let mut result = self.clone();
        result.shape = Shape::new(&axes.iter().map(|&i| self.shape()[i]).collect::<Vec<_>>());

        // Note: This doesn't actually reorder the data for ND tensors
        // A full implementation would need to handle data reordering
        Ok(result)
    }

    /// Adds a new dimension of size 1 at the specified axis
    pub fn expand_dims(&self, axis: usize) -> Result<Self, String> {
        trace_fn!("Tensor::expand_dims");
        debug!("Expanding dimensions at axis: {}", axis);

        let ndim = self.shape().len();
        if axis > ndim {
            let err = format!("Axis {axis} is out of bounds for tensor with {ndim} dimensions",);
            error!("{}", err);
            return Err(err);
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

        self.reshape(&new_shape)
    }

    /// Computes the sum of tensor elements along the specified axis
    /// If axis is None, sums all elements
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    pub fn sum(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::sum");
        debug!("Computing sum along axis: {:?}", axis);

        match axis {
            Some(ax) => {
                if ax >= self.shape().len() {
                    let err = format!("Axis {ax} is out of bounds for tensor dimensions");
                    error!("{}", err);
                    return Err(err);
                }

                let mut new_shape: Vec<usize> = self.shape().to_vec();
                new_shape[ax] = 1;

                // For now, implement a simple version that works for 1D and 2D
                // A more complete implementation would handle ND tensors
                if self.shape().len() == 1 {
                    let sum = self.data.iter().sum();
                    Tensor::from_vec(vec![sum], &[1], self.device)
                } else if self.shape().len() == 2 {
                    let (rows, cols) = (self.shape()[0], self.shape()[1]);

                    #[cfg(feature = "parallel")]
                    {
                        debug!("Using parallel sum along axis {}", ax);
                        let data = &self.data;

                        if ax == 0 {
                            // Sum along rows (axis 0)
                            let result: Vec<f32> = (0..cols)
                                .into_par_iter()
                                .map(|j| (0..rows).map(|i| data[i * cols + j]).sum())
                                .collect();
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Sum along columns (axis 1)
                            let result: Vec<f32> = (0..rows)
                                .into_par_iter()
                                .map(|i| data[i * cols..(i + 1) * cols].iter().sum())
                                .collect();
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }

                    #[cfg(not(feature = "parallel"))]
                    {
                        debug!("Using sequential sum along axis {}", ax);
                        if ax == 0 {
                            // Sum along rows (axis 0)
                            let mut result = vec![0.0; cols];
                            for i in 0..rows {
                                for (j, item) in result.iter_mut().enumerate().take(cols) {
                                    *item += self.data[i * cols + j];
                                }
                            }
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Sum along columns (axis 1)
                            let mut result = vec![0.0; rows];
                            for (i, item) in result.iter_mut().enumerate().take(rows) {
                                *item = self.data[i * cols..(i + 1) * cols].iter().sum();
                            }
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }
                } else {
                    // For higher dimensions, return a placeholder
                    // A full implementation would handle ND tensors
                    let err = "Sum for tensors with >2 dimensions not yet implemented".to_string();
                    error!("{}", err);
                    Err(err)
                }
            }
            None => {
                // Sum all elements
                #[cfg(feature = "parallel")]
                {
                    debug!("Using parallel global sum");
                    let sum = self.data.par_iter().sum();
                    Tensor::from_vec(vec![sum], &[1], self.device)
                }

                #[cfg(not(feature = "parallel"))]
                {
                    debug!("Using sequential global sum");
                    let sum = self.data.iter().sum();
                    Tensor::from_vec(vec![sum], &[1], self.device)
                }
            }
        }
    }

    /// Computes the mean of tensor elements along the specified axis
    /// If axis is None, computes mean of all elements
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::mean");
        debug!("Computing mean along axis: {:?}", axis);

        let sum = self.sum(axis)?;
        let count = match axis {
            Some(ax) => {
                if ax >= self.shape().len() {
                    return Err(format!("Axis {ax} out of bounds"));
                }
                self.shape()[ax] as f32
            }
            None => self.data.len() as f32,
        };

        // Divide each element by the count
        let mean_tensor = sum / count;
        Ok(mean_tensor)
    }

    /// Finds the maximum value along the specified axis
    /// If axis is None, finds the global maximum
    pub fn max(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::max");
        self.reduce_axis(axis, |a, b| a.max(*b), f32::NEG_INFINITY)
    }

    /// Finds the minimum value along the specified axis
    /// If axis is None, finds the global minimum
    pub fn min(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::min");
        self.reduce_axis(axis, |a, b| a.min(*b), f32::INFINITY)
    }

    /// Finds the index of the maximum value along the specified axis
    /// If axis is None, finds the index of the global maximum
    pub fn argmax(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmax");
        self.arg_reduce_axis(axis, |a, b| a > b)
    }

    /// Finds the index of the minimum value along the specified axis
    /// If axis is None, finds the index of the global minimum
    pub fn argmin(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmin");
        self.arg_reduce_axis(axis, |a, b| a < b)
    }

    /// Helper function for reduction operations along an axis
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn reduce_axis<F>(&self, axis: Option<usize>, reduce_op: F, init: f32) -> Result<Self, String>
    where
        F: Fn(f32, &f32) -> f32 + Send + Sync,
    {
        match axis {
            Some(ax) => {
                if ax >= self.shape().len() {
                    return Err(format!(
                        "Axis {ax} is out of bounds for tensor with {} dimensions",
                        self.shape().len()
                    ));
                }

                // For simplicity, implement for 1D and 2D tensors
                if self.shape().len() == 1 {
                    #[cfg(feature = "parallel")]
                    {
                        debug!("Using parallel 1D reduction");
                        let result = self
                            .data
                            .par_iter()
                            .fold(|| init, &reduce_op)
                            .reduce(|| init, |a, b| reduce_op(a, &b));
                        Tensor::from_vec(vec![result], &[1], self.device)
                    }

                    #[cfg(not(feature = "parallel"))]
                    {
                        debug!("Using sequential 1D reduction");
                        let result = self.data.iter().fold(init, &reduce_op);
                        Tensor::from_vec(vec![result], &[1], self.device)
                    }
                } else if self.shape().len() == 2 {
                    let (rows, cols) = (self.shape()[0], self.shape()[1]);

                    #[cfg(feature = "parallel")]
                    {
                        debug!("Using parallel 2D reduction along axis {}", ax);
                        let data = &self.data;

                        if ax == 0 {
                            // Parallel reduction along rows (axis 0)
                            let result: Vec<f32> = (0..cols)
                                .into_par_iter()
                                .map(|j| {
                                    (0..rows)
                                        .fold(init, |acc, i| reduce_op(acc, &data[i * cols + j]))
                                })
                                .collect();
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Parallel reduction along columns (axis 1)
                            let result: Vec<f32> = (0..rows)
                                .into_par_iter()
                                .map(|i| {
                                    data[i * cols..(i + 1) * cols].iter().fold(init, &reduce_op)
                                })
                                .collect();
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }

                    #[cfg(not(feature = "parallel"))]
                    {
                        debug!("Using sequential 2D reduction along axis {}", ax);
                        if ax == 0 {
                            // Reduce along rows (axis 0)
                            let mut result = vec![init; cols];
                            for i in 0..rows {
                                for (j, item) in result.iter_mut().enumerate().take(cols) {
                                    *item = reduce_op(*item, &self.data[i * cols + j]);
                                }
                            }
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Reduce along columns (axis 1)
                            let mut result = vec![init; rows];
                            for (i, item) in result.iter_mut().enumerate().take(rows) {
                                *item = self.data[i * cols..(i + 1) * cols]
                                    .iter()
                                    .fold(init, &reduce_op);
                            }
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }
                } else {
                    // For higher dimensions, return a placeholder
                    // A full implementation would handle ND tensors
                    Err("Reduction for tensors with >2 dimensions not yet implemented".to_string())
                }
            }
            None => {
                // Global reduction
                #[cfg(feature = "parallel")]
                {
                    debug!("Using parallel global reduction");
                    let result = self
                        .data
                        .par_iter()
                        .fold(|| init, &reduce_op)
                        .reduce(|| init, |a, b| reduce_op(a, &b));
                    Tensor::from_vec(vec![result], &[1], self.device)
                }

                #[cfg(not(feature = "parallel"))]
                {
                    debug!("Using sequential global reduction");
                    let result = self.data.iter().fold(init, &reduce_op);
                    Tensor::from_vec(vec![result], &[1], self.device)
                }
            }
        }
    }

    /// Helper function for arg reduction operations along an axis
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn arg_reduce_axis<F>(&self, axis: Option<usize>, compare: F) -> Result<Self, String>
    where
        F: Fn(f32, f32) -> bool + Send + Sync,
    {
        match axis {
            Some(ax) => {
                if ax >= self.shape().len() {
                    return Err(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        ax,
                        self.shape().len()
                    ));
                }

                // For simplicity, implement for 1D and 2D tensors
                if self.shape().len() == 1 {
                    #[cfg(feature = "parallel")]
                    {
                        debug!("Using parallel 1D arg reduction");
                        let (idx, _) = self
                            .data
                            .par_iter()
                            .enumerate()
                            .fold(
                                || (0, &self.data[0]),
                                |(max_idx, max_val), (i, val)| {
                                    if compare(*val, *max_val) {
                                        (i, val)
                                    } else {
                                        (max_idx, max_val)
                                    }
                                },
                            )
                            .reduce(
                                || (0, &self.data[0]),
                                |a, b| {
                                    if compare(*b.1, *a.1) {
                                        b
                                    } else {
                                        a
                                    }
                                },
                            );
                        Tensor::from_vec(vec![idx as f32], &[1], self.device)
                    }

                    #[cfg(not(feature = "parallel"))]
                    {
                        debug!("Using sequential 1D arg reduction");
                        let (idx, _) = self.data.iter().enumerate().fold(
                            (0, &self.data[0]),
                            |(max_idx, max_val), (i, val)| {
                                if compare(*val, *max_val) {
                                    (i, val)
                                } else {
                                    (max_idx, max_val)
                                }
                            },
                        );
                        Tensor::from_vec(vec![idx as f32], &[1], self.device)
                    }
                } else if self.shape().len() == 2 {
                    let (rows, cols) = (self.shape()[0], self.shape()[1]);

                    #[cfg(feature = "parallel")]
                    {
                        debug!("Using parallel 2D arg reduction along axis {}", ax);
                        let data = &self.data;

                        if ax == 0 {
                            // Parallel arg reduction along rows (axis 0)
                            let result: Vec<f32> = (0..cols)
                                .into_par_iter()
                                .map(|j| {
                                    let mut max_idx = 0;
                                    let mut max_val = data[j];
                                    for i in 1..rows {
                                        let val = data[i * cols + j];
                                        if compare(val, max_val) {
                                            max_val = val;
                                            max_idx = i;
                                        }
                                    }
                                    max_idx as f32
                                })
                                .collect();
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Parallel arg reduction along columns (axis 1)
                            let result: Vec<f32> = (0..rows)
                                .into_par_iter()
                                .map(|i| {
                                    let start = i * cols;
                                    let end = start + cols;
                                    let (idx, _) = data[start..end].iter().enumerate().fold(
                                        (0, &data[start]),
                                        |(max_idx, max_val), (j, val)| {
                                            if compare(*val, *max_val) {
                                                (j, val)
                                            } else {
                                                (max_idx, max_val)
                                            }
                                        },
                                    );
                                    idx as f32
                                })
                                .collect();
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }

                    #[cfg(not(feature = "parallel"))]
                    {
                        debug!("Using sequential 2D arg reduction along axis {}", ax);
                        if ax == 0 {
                            // Arg reduce along rows (axis 0)
                            let mut result = vec![0.0; cols];
                            for (j, item) in result.iter_mut().enumerate().take(cols) {
                                let mut max_idx = 0;
                                let mut max_val = self.data[j];
                                for i in 1..rows {
                                    let val = self.data[i * cols + j];
                                    if compare(val, max_val) {
                                        max_val = val;
                                        max_idx = i;
                                    }
                                }
                                *item = max_idx as f32;
                            }
                            Tensor::from_vec(result, &[1, cols], self.device)
                        } else {
                            // Arg reduce along columns (axis 1)
                            let mut result = vec![0.0; rows];
                            for (i, item) in result.iter_mut().enumerate().take(rows) {
                                let start = i * cols;
                                let end = start + cols;
                                let (idx, _) = self.data[start..end].iter().enumerate().fold(
                                    (0, &self.data[start]),
                                    |(max_idx, max_val), (j, val)| {
                                        if compare(*val, *max_val) {
                                            (j, val)
                                        } else {
                                            (max_idx, max_val)
                                        }
                                    },
                                );
                                *item = idx as f32;
                            }
                            Tensor::from_vec(result, &[rows, 1], self.device)
                        }
                    }
                } else {
                    // For higher dimensions, return a placeholder
                    // A full implementation would handle ND tensors
                    Err(
                        "Arg reduction for tensors with >2 dimensions not yet implemented"
                            .to_string(),
                    )
                }
            }
            None => {
                // Global arg reduction
                #[cfg(feature = "parallel")]
                {
                    debug!("Using parallel global arg reduction");
                    let (idx, _) = self
                        .data
                        .par_iter()
                        .enumerate()
                        .fold(
                            || (0, &self.data[0]),
                            |(max_idx, max_val), (i, val)| {
                                if compare(*val, *max_val) {
                                    (i, val)
                                } else {
                                    (max_idx, max_val)
                                }
                            },
                        )
                        .reduce(
                            || (0, &self.data[0]),
                            |a, b| {
                                if compare(*b.1, *a.1) {
                                    b
                                } else {
                                    a
                                }
                            },
                        );
                    Tensor::from_vec(vec![idx as f32], &[1], self.device)
                }

                #[cfg(not(feature = "parallel"))]
                {
                    debug!("Using sequential global arg reduction");
                    let (idx, _) = self.data.iter().enumerate().fold(
                        (0, &self.data[0]),
                        |(max_idx, max_val), (i, val)| {
                            if compare(*val, *max_val) {
                                (i, val)
                            } else {
                                (max_idx, max_val)
                            }
                        },
                    );
                    Tensor::from_vec(vec![idx as f32], &[1], self.device)
                }
            }
        }
    }

    /// Removes dimensions of size 1
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    pub fn squeeze(&self, axis: Option<usize>) -> Self {
        trace_fn!("Tensor::squeeze");
        debug!(
            "Squeezing tensor with shape: {:?}, axis: {:?}",
            self.shape(),
            axis
        );

        let new_shape: Vec<usize> = match axis {
            Some(ax) => {
                if ax >= self.shape().len() || self.shape()[ax] != 1 {
                    // If axis is out of bounds or dimension is not 1, return original
                    debug!("No dimensions to squeeze, returning original tensor");
                    return self.clone();
                }
                self.shape()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| if i != ax { Some(d) } else { None })
                    .collect()
            }
            None => {
                // Remove all dimensions of size 1
                self.shape().iter().filter(|&&d| d != 1).cloned().collect()
            }
        };

        // If all dimensions are removed, return a scalar in a 1D tensor
        if new_shape.is_empty() {
            debug!("All dimensions squeezed, returning scalar in 1D tensor");
            return Tensor::from_vec(vec![self.data[0]], &[1], self.device).unwrap_or_else(|_| {
                error!("Failed to create scalar tensor, returning original");
                self.clone()
            });
        }

        // Reshape the tensor to the new shape
        match self.reshape(&new_shape) {
            Ok(tensor) => {
                debug!(
                    "Successfully squeezed tensor to shape: {:?}",
                    tensor.shape()
                );
                tensor
            }
            Err(e) => {
                error!("Failed to reshape tensor during squeeze: {}", e);
                self.clone() // Return original if reshape fails
            }
        }
    }

    /// Converts the tensor to a vector
    pub fn to_vec(&self) -> Vec<f32> {
        trace_fn!("Tensor::to_vec");
        debug!("Converting tensor with shape {:?} to Vec", self.shape());
        self.data.clone()
    }

    /// Moves the tensor to the specified device
    pub fn to_device(&self, device: Device) -> Result<Self, String> {
        trace_fn!("Tensor::to_device");
        debug!("Moving tensor from {:?} to {:?}", self.device, device);

        // For now, just clone the data since we only support CPU
        // In a real implementation, this would handle device transfers
        if matches!(self.device, Device::Cpu(_)) && matches!(device, Device::Cpu(_)) {
            debug!("Device transfer completed (CPU to CPU)");
            Ok(self.clone())
        } else {
            let err = "Device transfer not implemented yet".to_string();
            error!("{}", err);
            Err(err)
        }
    }

    /// Performs matrix multiplication between two tensors.
    /// Supports 1D and 2D tensors only.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        trace_fn!("Tensor::matmul");
        debug!(
            "Matrix multiplication between shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );

        // Check if both tensors are 1D or 2D
        if self.shape().len() > 2 || other.shape().len() > 2 {
            return Err(
                "Matrix multiplication is only supported for 1D and 2D tensors".to_string(),
            );
        }

        // Handle 1D x 1D (dot product)
        if self.shape().len() == 1 && other.shape().len() == 1 {
            if self.shape()[0] != other.shape()[0] {
                let err = "Incompatible shapes for dot product".to_string();
                error!("{}: {:?} and {:?}", err, self.shape(), other.shape());
                return Err(err);
            }

            #[cfg(feature = "parallel")]
            {
                // TODO : map vs fold in not parallel version
                debug!("Using parallel dot product");
                let dot_product = self
                    .data
                    .par_iter()
                    .zip(&other.data)
                    .map(|(&a, &b)| a * b)
                    .sum();
                return Tensor::from_vec(vec![dot_product], &[1], self.device);
            }

            #[cfg(not(feature = "parallel"))]
            {
                debug!("Using sequential dot product");
                let dot_product = self
                    .data
                    .iter()
                    .zip(&other.data)
                    .fold(0.0, |acc, (&a, &b)| acc + a * b);
                return Tensor::from_vec(vec![dot_product], &[1], self.device);
            }
        }

        // Handle 2D x 2D or 2D x 1D matrix multiplication
        let (m, k1) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = if other.shape().len() == 2 {
            (other.shape()[0], other.shape()[1])
        } else {
            (other.shape()[0], 1)
        };

        debug!("Matrix multiplication: ({}, {}) x ({}, {})", m, k1, k2, n);

        if k1 != k2 {
            let err = format!(
                "Incompatible shapes for matrix multiplication: {:?} and {:?}",
                self.shape(),
                other.shape()
            );
            error!("{}", err);
            return Err(err);
        }

        let mut result_data = vec![0.0; m * n];
        let other_ndim = other.shape().len();

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel matrix multiplication ({}x{} * {}x{})",
                m, k1, k2, n
            );
            let a_data = &self.data;
            let b_data = &other.data;

            result_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    for (j, item) in row.iter_mut().enumerate().take(n) {
                        let mut sum = 0.0;
                        for k in 0..k1 {
                            let a_idx = i * k1 + k;
                            let b_idx = if other_ndim == 2 { k * n + j } else { k };
                            sum += a_data[a_idx] * b_data[b_idx];
                        }
                        *item = sum;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential matrix multiplication ({}x{} * {}x{})",
                m, k1, k2, n
            );
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        let a_idx = i * k1 + k;
                        let b_idx = if other_ndim == 2 { k * n + j } else { k };
                        sum += self.data[a_idx] * other.data[b_idx];
                    }
                    result_data[i * n + j] = sum;
                }
            }
        }

        let result_shape = if other.shape().len() == 2 {
            vec![m, n]
        } else {
            vec![m]
        };

        let result = Tensor::from_vec(result_data, &result_shape, self.device)?;
        debug!("Matrix multiplication result shape: {:?}", result.shape());
        Ok(result)
    }

    /// Applies the ReLU activation function element-wise.
    /// This operation is parallelized for tensors larger than MIN_PARALLEL_SIZE
    /// when the 'parallel' feature is enabled.
    pub fn relu(&self) -> Result<Tensor, String> {
        trace_fn!("Tensor::relu");
        debug!(
            "Applying ReLU activation to tensor with shape {:?}",
            self.shape()
        );

        let mut result = self.clone();

        #[cfg(feature = "parallel")]
        {
            debug!("Using parallel ReLU ({} elements)", result.data.len());
            result.data.par_iter_mut().for_each(|x| {
                *x = x.max(0.0);
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!("Using sequential ReLU ({} elements)", result.data.len());
            for x in &mut result.data {
                *x = x.max(0.0);
            }
        }

        Ok(result)
    }

    /// Element-wise addition with another tensor.
    pub fn add_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        trace_fn!("Tensor::add_tensor");
        debug!(
            "Adding tensors with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        self.binary_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction with another tensor.
    pub fn sub_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        trace_fn!("Tensor::sub_tensor");
        debug!(
            "Subtracting tensors with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        self.binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication with another tensor.
    pub fn mul_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        trace_fn!("Tensor::mul_tensor");
        debug!(
            "Element-wise multiplication of tensors with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        self.binary_op(other, |a, b| a * b)
    }

    /// Element-wise division by another tensor.
    pub fn div_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        trace_fn!("Tensor::div_tensor");
        debug!(
            "Element-wise division of tensors with shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );

        self.binary_op(other, |a, b| {
            if b == 0.0 {
                if a == 0.0 {
                    0.0
                } else {
                    f32::INFINITY
                }
            } else {
                a / b
            }
        })
    }

    /// Helper function for binary operations with broadcasting support.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn binary_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, String>
    where
        F: Fn(f32, f32) -> f32 + Send + Sync,
    {
        trace_fn!("Tensor::binary_op");
        debug!(
            "Binary operation between shapes {:?} and {:?}",
            self.shape(),
            other.shape()
        );

        // For now, only support tensors with the same shape
        if self.shape() != other.shape() {
            return Err(format!(
                "Incompatible shapes for binary operation: {:?} and {:?}",
                self.shape(),
                other.shape()
            ));
        }

        debug!("Performing element-wise operation between tensors");
        let len = self.data.len();
        let mut result_data = vec![0.0; len];

        #[cfg(feature = "parallel")]
        {
            debug!("Using parallel binary operation ({} elements)", len);
            result_data
                .par_iter_mut()
                .zip(self.data.par_iter().zip(other.data.par_iter()))
                .for_each(|(r, (&a, &b))| {
                    *r = op(a, b);
                });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        debug!("Using sequential binary operation ({} elements)", len);
        result_data
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
            .for_each(|(r, (&a, &b))| {
                *r = op(a, b);
            });

        debug!("Binary operation completed successfully");
        Tensor::from_vec(result_data, self.shape(), self.device)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
        }
    }
}

// Implement basic arithmetic operations with scalar values
impl Add<f32> for Tensor {
    type Output = Tensor;

    /// Adds a scalar to each element of the tensor.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn add(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::add_scalar");
        debug!(
            "Adding scalar {} to tensor with shape {:?}",
            rhs,
            self.shape()
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel scalar addition ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x += rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential scalar addition ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x += rhs;
            }
        }

        debug!("Scalar addition completed");
        self
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    /// Subtracts a scalar from each element of the tensor.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn sub(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::sub_scalar");
        debug!(
            "Subtracting scalar {} from tensor with shape {:?}",
            rhs,
            self.shape()
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel scalar subtraction ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x -= rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential scalar subtraction ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x -= rhs;
            }
        }

        debug!("Scalar subtraction completed");
        self
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    /// Multiplies each element of the tensor by a scalar.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn mul(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::mul_scalar");
        debug!(
            "Multiplying tensor with shape {:?} by scalar {}",
            self.shape(),
            rhs
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel scalar multiplication ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x *= rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential scalar multiplication ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x *= rhs;
            }
        }

        debug!("Scalar multiplication completed");
        self
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    /// Divides each element of the tensor by a scalar.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    ///
    /// # Panics
    /// This function will not panic on division by zero, but will return
    /// positive infinity for all elements in the tensor.
    fn div(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::div_scalar");
        debug!(
            "Dividing tensor with shape {:?} by scalar {}",
            self.shape(),
            rhs
        );

        if rhs == 0.0 {
            error!("Attempted division by zero, returning infinity");
            self.data = vec![f32::INFINITY; self.shape.len()];
            return self;
        }
        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel scalar multiplication ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x /= rhs;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for x in &mut self.data {
                *x /= rhs;
            }
        }

        debug!("Scalar division completed");
        self
    }
}

impl AddAssign<f32> for Tensor {
    /// Adds a scalar to each element of the tensor in-place.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn add_assign(&mut self, rhs: f32) {
        trace_fn!("Tensor::add_assign");
        debug!(
            "Adding scalar {} in-place to tensor with shape {:?}",
            rhs,
            self.shape()
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel in-place addition ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x += rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential in-place addition ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x += rhs;
            }
        }

        debug!("In-place scalar addition completed");
    }
}

impl SubAssign<f32> for Tensor {
    /// Subtracts a scalar from each element of the tensor in-place.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn sub_assign(&mut self, rhs: f32) {
        trace_fn!("Tensor::sub_assign");
        debug!(
            "Subtracting scalar {} in-place from tensor with shape {:?}",
            rhs,
            self.shape()
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel in-place subtraction ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x -= rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential in-place subtraction ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x -= rhs;
            }
        }

        debug!("In-place scalar subtraction completed");
    }
}

impl MulAssign<f32> for Tensor {
    /// Multiplies each element of the tensor by a scalar in-place.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn mul_assign(&mut self, rhs: f32) {
        trace_fn!("Tensor::mul_assign");
        debug!(
            "Multiplying tensor in-place with shape {:?} by scalar {}",
            self.shape(),
            rhs
        );

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel in-place multiplication ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x *= rhs;
            });
        }

        // Sequential implementation (used when parallel is disabled or for small tensors)
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential in-place multiplication ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x *= rhs;
            }
        }

        debug!("In-place scalar multiplication completed");
    }
}

impl DivAssign<f32> for Tensor {
    /// Divides each element of the tensor by a scalar in-place.
    ///
    /// # Note
    /// When the 'parallel' feature is enabled, operations on large tensors
    /// will be parallelized for better performance.
    fn div_assign(&mut self, rhs: f32) {
        trace_fn!("Tensor::div_assign");
        debug!(
            "Dividing tensor in-place with shape {:?} by scalar {}",
            self.shape(),
            rhs
        );

        if rhs == 0.0 {
            error!("Attempted in-place division by zero, returning infinity");
            self.data = vec![f32::INFINITY; self.shape.len()];
            return;
        }

        #[cfg(feature = "parallel")]
        {
            debug!(
                "Using parallel in-place division by zero ({} elements)",
                self.data.len()
            );
            self.data.par_iter_mut().for_each(|x| {
                *x /= rhs;
            });
        }

        // Sequential implementation for division by zero
        #[cfg(not(feature = "parallel"))]
        {
            debug!(
                "Using sequential in-place division by zero ({} elements)",
                self.data.len()
            );
            for x in &mut self.data {
                *x /= rhs;
            }
        }

        debug!("In-place scalar division completed");
    }
}
// Implement display for better debugging
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, device={:?}, dtype={:?})",
            self.shape.dims(),
            self.device,
            self.dtype
        )
    }
}

#[allow(unused_imports)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Test tensor creation and basic properties
    #[test]
    fn test_tensor_creation() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_creation_invalid_shape() {
        let result = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2], Device::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_ones_zeros() {
        let ones = Tensor::ones(&[2, 2], Device::default());
        assert_eq!(ones.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);

        let zeros = Tensor::zeros(&[2, 2], Device::default());
        assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_identity() {
        let identity = Tensor::identity(3, Device::default());
        assert_eq!(identity.shape(), &[3, 3]);
        assert_eq!(
            identity.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_random() {
        let random = Tensor::random(&[10, 10], Device::default());
        assert_eq!(random.shape(), &[10, 10]);
        assert!(random.to_vec().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_arange() {
        let range = Tensor::arange(0.0, 5.0);
        assert_eq!(range.shape(), &[5]);
        assert_eq!(range.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[2, 2], Device::default()).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], Device::default()).unwrap();
        let reshaped = tensor.reshape(&[2, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 2]);
        assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Test invalid reshape
        assert!(tensor.reshape(&[3, 2]).is_err());
    }

    #[test]
    fn test_transpose() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let transposed = tensor.transpose(None).unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_expand_dims() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], Device::default()).unwrap();
        let expanded = tensor.expand_dims(0).unwrap();
        assert_eq!(expanded.shape(), &[1, 4]);

        let expanded2 = tensor.expand_dims(1).unwrap();
        assert_eq!(expanded2.shape(), &[4, 1]);
    }

    #[test]
    fn test_squeeze() {
        let tensor = Tensor::from_vec(vec![1.0], &[1, 1, 1], Device::default()).unwrap();
        let squeezed = tensor.squeeze(None);
        assert_eq!(squeezed.shape(), &[1]);

        let tensor2 = Tensor::from_vec(vec![1.0, 2.0], &[2, 1], Device::default()).unwrap();
        let squeezed2 = tensor2.squeeze(Some(1));
        assert_eq!(squeezed2.shape(), &[2]);
    }

    #[test]
    fn test_reduction_ops() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();

        // Test sum
        let sum_all = tensor.sum(None).unwrap();
        assert_relative_eq!(sum_all.to_vec()[0], 10.0);

        let sum_axis0 = tensor.sum(Some(0)).unwrap();
        assert_eq!(sum_axis0.shape(), &[1, 2]);
        assert_eq!(sum_axis0.to_vec(), vec![4.0, 6.0]);

        // Test mean
        let mean_all = tensor.mean(None).unwrap();
        assert_relative_eq!(mean_all.to_vec()[0], 2.5);

        // Test max/min
        let max_val = tensor.max(None).unwrap();
        assert_relative_eq!(max_val.to_vec()[0], 4.0);

        let min_val = tensor.min(None).unwrap();
        assert_relative_eq!(min_val.to_vec()[0], 1.0);

        // Test argmax/argmin
        let argmax = tensor.argmax(None).unwrap();
        assert_eq!(argmax.to_vec()[0] as usize, 3);

        let argmin = tensor.argmin(None).unwrap();
        assert_eq!(argmin.to_vec()[0] as usize, 0);
    }

    #[test]
    fn test_shape_calculation() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.len(), 24);
        assert!(!shape.is_empty());
    }

    #[test]
    fn test_dtype_conversion() {
        let dtype: DType = "f32".try_into().unwrap();
        assert_eq!(dtype, DType::F32);

        let dtype_str: &str = DType::F32.try_into().unwrap();
        assert_eq!(dtype_str, "f32");
    }

    #[test]
    fn test_device_handling() {
        let tensor = Tensor::from_vec(vec![1.0], &[1], Device::default()).unwrap();
        assert_eq!(tensor.device(), &Device::Cpu(Some(0)));

        let moved = tensor.to_device(Device::Cpu(Some(0))).unwrap();
        assert_eq!(moved.device(), &Device::Cpu(Some(0)));

        // TODO : more to come ?
    }

    #[test]
    fn test_relu() {
        let tensor =
            Tensor::from_vec(vec![-1.0, 0.0, 1.0, -2.0], &[2, 2], Device::default()).unwrap();
        let relu = tensor.relu().unwrap();
        assert_eq!(relu.to_vec(), vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_scalar_addition() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let result = tensor + 1.0;
        assert_eq!(result.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor_clone() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let cloned = tensor.clone();
        assert_eq!(tensor.to_vec(), cloned.to_vec());
        assert_eq!(tensor.shape(), cloned.shape());
    }

    #[test]
    fn test_tensor_debug() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("shape=[2, 2]"));
        assert!(debug_str.contains("device=Cpu(Some(0))"));
        assert!(debug_str.contains("dtype=F32"));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::default()).unwrap();

        // Test matrix multiplication
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);

        // Test invalid dimensions
        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], Device::default()).unwrap();
        assert!(a.matmul(&d).is_err());
    }

    #[test]
    fn test_elementwise_operations() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[2, 2], Device::default()).unwrap();

        // Test addition
        let c = a.add_tensor(&b).unwrap();
        assert_eq!(c.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);

        // Test subtraction
        let d = a.sub_tensor(&b).unwrap();
        assert_eq!(d.to_vec(), vec![-1.0, 0.0, 1.0, 2.0]);

        // Test multiplication
        let e = a.mul_tensor(&b).unwrap();
        assert_eq!(e.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

        // Test division
        let f = b.div_tensor(&a).unwrap();
        assert_eq!(f.to_vec(), vec![2.0, 1.0, 0.6666667, 0.5]);
    }

    #[test]
    fn test_scalar_operations() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();

        // Test addition
        let b = a.clone() + 1.0;
        assert_eq!(b.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);

        // Test subtraction
        let c = a.clone() - 1.0;
        assert_eq!(c.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);

        // Test multiplication
        let d = a.clone() * 2.0;
        assert_eq!(d.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

        // Test division
        let e = a.clone() / 2.0;
        assert_eq!(e.to_vec(), vec![0.5, 1.0, 1.5, 2.0]);

        let c = b - 1.0;
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Test scalar multiplication
        let d = c * 2.0;
        assert_eq!(d.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

        // Test scalar division
        let e = d / 2.0;
        assert_eq!(e.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Test division by zero
        let f = e / 0.0;
        assert_eq!(
            f.to_vec(),
            vec![f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY]
        );

        // Test in-place operations
        a += 1.0;
        assert_eq!(a.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);

        a -= 1.0;
        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        a *= 2.0;

        assert_eq!(a.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
        a /= 2.0;

        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Test in-place division by zero
        a /= 0.0;
        assert_eq!(
            a.to_vec(),
            vec![f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY]
        );
    }

    #[test]
    fn test_invalid_operations() {
        // Test unsupported 3D tensors
        let t1 = Tensor::ones(&[2, 3, 4], Device::default());
        let t2 = Tensor::ones(&[4, 5, 6], Device::default());
        assert!(t1.matmul(&t2).is_err());

        // Different devices (once implemented)
        // let t3 = Tensor::ones(&[2, 2], Device::Cuda(0));
        // assert!(t1.add(&t3).is_err());
    }
}
