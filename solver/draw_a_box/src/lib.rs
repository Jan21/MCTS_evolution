/// Weight of a box-drawing character segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    Empty,
    Light,
    Heavy,
}

/// Find a box-drawing character for the given weights.
/// This is a stub implementation that returns a space for all inputs.
pub fn find_character(_up: Weight, _right: Weight, _down: Weight, _left: Weight) -> &'static str {
    " "
}
