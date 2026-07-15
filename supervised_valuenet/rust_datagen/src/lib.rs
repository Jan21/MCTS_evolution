//! rust_datagen — solving/labeling library for Ricochet Robots training data.
//!
//! See DESIGN.md for the work-item schema, exact-parity semantics, and the
//! gate/test contract. Python (`supervised_valuenet/`) is the reference
//! implementation; every public function here names the Python code it ports.

pub mod types;
pub mod physics;
pub mod board;
pub mod move_oracle;
pub mod subgoal;
pub mod io;
