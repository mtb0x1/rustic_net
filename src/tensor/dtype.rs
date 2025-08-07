use crate::trace_fn;
use tracing::debug;

/// Numeric type of tensor elements.
///
/// Defines the data type of elements stored in a tensor. Currently,
/// only 32-bit floating point (f32) is supported, but the enum is
/// designed to be extended with additional types in the future.
///
/// # Examples
/// ```rust
/// use rustic_net::tensor::DType;
///
/// let dtype = DType::F32;
/// assert_eq!(dtype.size_of(), 4);  // 4 bytes for f32
/// ```
///
/// # Type Safety
/// The `DType` enum ensures type safety by preventing incompatible operations
/// between tensors of different data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point number (IEEE 754)
    ///
    /// - Size: 4 bytes
    /// - Range: Approximately ±3.4 × 10^38 with 7 decimal digits of precision
    F32,
    // Future types:
    // F64,
    // I32,
    // I64,
    // U8,
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
