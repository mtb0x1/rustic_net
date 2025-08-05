use rustic_net::tensor::Shape;

#[test]
fn test_shape_creation() {
    let shape = Shape::new(&[2, 3, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);
    assert_eq!(shape.strides(), &[12, 4, 1]);
    assert_eq!(shape.len(), 24);
    assert_eq!(shape.ndim(), 3);
    assert!(!shape.is_empty());
}

#[test]
fn test_shape_1d() {
    let shape = Shape::new(&[5]);
    assert_eq!(shape.dims(), &[5]);
    assert_eq!(shape.strides(), &[1]);
    assert_eq!(shape.len(), 5);
    assert_eq!(shape.ndim(), 1);
    assert!(!shape.is_empty());
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
