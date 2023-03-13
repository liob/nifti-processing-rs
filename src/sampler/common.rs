/// A set of strategies a sampler may employ if a point is out of sample.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    Constant,
    Nearest,
}
