use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ricochet_board::generator::Generator;
use ricochet_board::{
    quadrant, Board, Direction, Field, Position, PositionEncoding, Robot, RobotPositions, Round,
    Symbol, Target,
};
use ricochet_solver::{BreadthFirst, Solver};

#[pymodule]
fn ricochet_solver_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_round, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_walls, m)?)?;
    m.add_function(wrap_pyfunction!(solve_generated, m)?)?;
    Ok(())
}

/// Solve a standard 16x16 board round identified by a seed (0..8262).
///
/// Args:
///     round_seed: int in [0, 8262)
///     robot_positions: list of 4 (col, row) tuples [red, blue, green, yellow]
///     target_robot: optional override - "red", "blue", "green", "yellow", or "any".
///         If None, uses the target baked into the seed.
///     target_pos: optional (col, row) override for the target position.
///         If None, uses the position baked into the seed.
///
/// Returns:
///     dict with "moves" (list of (robot, direction) string tuples),
///     "move_count" (int), "start_positions" and "end_positions".
#[pyfunction]
#[pyo3(signature = (round_seed, robot_positions, target_robot=None, target_pos=None))]
fn solve_round(
    py: Python,
    round_seed: usize,
    robot_positions: Vec<(u16, u16)>,
    target_robot: Option<&str>,
    target_pos: Option<(u16, u16)>,
) -> PyResult<PyObject> {
    let positions = parse_positions(&robot_positions)?;
    let base_round = quadrant::round_from_seed(round_seed);

    let target = match target_robot {
        Some(s) => parse_target_from_robot(s)?,
        None => base_round.target(),
    };
    let target_position = match target_pos {
        Some((c, r)) => Position::new(c, r),
        None => base_round.target_position(),
    };
    let round = Round::new(base_round.board().clone(), target, target_position);

    do_solve(py, &round, positions)
}

/// Solve a board defined by explicit wall data.
///
/// Args:
///     walls_right: list of (col, row) tuples where a wall is to the right of the field
///     walls_down: list of (col, row) tuples where a wall is below the field
///     board_size: side length of the board
///     target_pos: (col, row) of the target
///     target_robot: "red", "blue", "green", "yellow", or "any"
///     robot_positions: list of 4 (col, row) tuples [red, blue, green, yellow]
///
/// Returns:
///     dict with "moves", "move_count", "start_positions", "end_positions".
#[pyfunction]
fn solve_with_walls(
    py: Python,
    walls_right: Vec<(u16, u16)>,
    walls_down: Vec<(u16, u16)>,
    board_size: u16,
    target_pos: (u16, u16),
    target_robot: &str,
    robot_positions: Vec<(u16, u16)>,
) -> PyResult<PyObject> {
    let positions = parse_positions(&robot_positions)?;

    let mut walls: Vec<Vec<Field>> = vec![vec![Field::default(); board_size as usize]; board_size as usize];
    for (col, row) in &walls_right {
        walls[*col as usize][*row as usize].right = true;
    }
    for (col, row) in &walls_down {
        walls[*col as usize][*row as usize].down = true;
    }
    let board = Board::new(walls);

    let target = parse_target_from_robot(target_robot)?;
    let target_position = Position::new(target_pos.0, target_pos.1);
    let round = Round::new(board, target, target_position);

    do_solve(py, &round, positions)
}

/// Solve a randomly generated board.
///
/// Args:
///     seed: int used to seed board generation
///     board_size: side length of the board
///     target_pos: (col, row) of the target
///     target_robot: "red", "blue", "green", "yellow", or "any"
///     robot_positions: list of 4 (col, row) tuples [red, blue, green, yellow]
///
/// Returns:
///     dict with "moves", "move_count", "start_positions", "end_positions".
#[pyfunction]
fn solve_generated(
    py: Python,
    seed: u128,
    board_size: u16,
    target_pos: (u16, u16),
    target_robot: &str,
    robot_positions: Vec<(u16, u16)>,
) -> PyResult<PyObject> {
    let positions = parse_positions(&robot_positions)?;

    let mut gen = Generator::from_seed(seed, board_size);
    let board = gen.generate_board();

    let target = parse_target_from_robot(target_robot)?;
    let target_position = Position::new(target_pos.0, target_pos.1);
    let round = Round::new(board, target, target_position);

    do_solve(py, &round, positions)
}

fn do_solve(py: Python, round: &Round, positions: RobotPositions) -> PyResult<PyObject> {
    let path = BreadthFirst::new().solve(round, positions);

    let moves: Vec<(&str, &str)> = path
        .movements()
        .iter()
        .map(|(robot, dir)| (robot_to_str(*robot), direction_to_str(*dir)))
        .collect();

    let start_pos: Vec<(u16, u16)> = path
        .start_pos()
        .to_array()
        .iter()
        .map(|p| (p.column(), p.row()))
        .collect();

    let end_pos: Vec<(u16, u16)> = path
        .end_pos()
        .to_array()
        .iter()
        .map(|p| (p.column(), p.row()))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("moves", moves)?;
    dict.set_item("move_count", path.len())?;
    dict.set_item("start_positions", start_pos)?;
    dict.set_item("end_positions", end_pos)?;
    Ok(dict.into())
}

fn parse_positions(pos: &[(u16, u16)]) -> PyResult<RobotPositions> {
    if pos.len() != 4 {
        return Err(PyValueError::new_err(
            "robot_positions must have exactly 4 entries: [red, blue, green, yellow]",
        ));
    }
    let arr: [(PositionEncoding, PositionEncoding); 4] = [pos[0], pos[1], pos[2], pos[3]];
    Ok(RobotPositions::from_tuples(&arr))
}

fn parse_target_from_robot(s: &str) -> PyResult<Target> {
    match s.to_lowercase().as_str() {
        "red" => Ok(Target::Red(Symbol::Circle)),
        "blue" => Ok(Target::Blue(Symbol::Circle)),
        "green" => Ok(Target::Green(Symbol::Circle)),
        "yellow" => Ok(Target::Yellow(Symbol::Circle)),
        "any" | "spiral" => Ok(Target::Spiral),
        _ => Err(PyValueError::new_err(format!(
            "Unknown target_robot '{}'. Use: red, blue, green, yellow, any",
            s
        ))),
    }
}

fn robot_to_str(r: Robot) -> &'static str {
    match r {
        Robot::Red => "red",
        Robot::Blue => "blue",
        Robot::Green => "green",
        Robot::Yellow => "yellow",
    }
}

fn direction_to_str(d: Direction) -> &'static str {
    match d {
        Direction::Up => "up",
        Direction::Down => "down",
        Direction::Right => "right",
        Direction::Left => "left",
    }
}
