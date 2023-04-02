/// A set of strategies a sampler may employ if a point is out of sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMode {
    /// The input is expanded by replacing all numbers outside of the edge 
    /// with the same constant value determined by the cval parameter.
    Constant,

    /// The nearest pixel value is duplicated to expand the input.
    Nearest,
}
