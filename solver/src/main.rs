use std::io::{self, Read};
use std::time::Instant;
use std::panic;

use ricochet_board::{Board, Field, Position, RobotPositions, Round, Target, Symbol, Robot, Direction};
use ricochet_solver::{AStar, Solver};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Instance {
    down_walls: Vec<Vec<bool>>,   // [col][row]: wall below cell
    right_walls: Vec<Vec<bool>>,  // [col][row]: wall right of cell
    robots: Vec<Vec<u16>>,        // [[col, row]; 4] for Red, Blue, Green, Yellow
    target: Vec<u16>,             // [col, row]
    #[allow(dead_code)]
    target_robot: u8,             // 0 = Red (always)
}

#[derive(Serialize)]
struct Move {
    robot: String,
    direction: String,
    from: [u16; 2],
    to: [u16; 2],
}

#[derive(Serialize)]
struct Result {
    solved: bool,
    moves: usize,
    path: Vec<Move>,
    time_us: u128,
    error: Option<String>,
}

fn robot_name(r: Robot) -> String {
    format!("{:?}", r)
}

fn direction_name(d: Direction) -> String {
    format!("{:?}", d)
}

fn solve_instance(inst: &Instance) -> Result {
    let size = 16usize;

    // Build Walls: walls[col][row] with .down and .right
    let mut walls: Vec<Vec<Field>> = vec![vec![Field::default(); size]; size];
    for col in 0..size {
        for row in 0..size {
            if col < inst.down_walls.len() && row < inst.down_walls[col].len() {
                walls[col][row].down = inst.down_walls[col][row];
            }
            if col < inst.right_walls.len() && row < inst.right_walls[col].len() {
                walls[col][row].right = inst.right_walls[col][row];
            }
        }
    }

    let board = Board::new(walls).wall_enclosure();

    // Robot positions: Red, Blue, Green, Yellow
    let positions_arr: [(u16, u16); 4] = [
        (inst.robots[0][0], inst.robots[0][1]),
        (inst.robots[1][0], inst.robots[1][1]),
        (inst.robots[2][0], inst.robots[2][1]),
        (inst.robots[3][0], inst.robots[3][1]),
    ];
    let start_positions = RobotPositions::from_tuples(&positions_arr);

    // Target position and target (always Red)
    let target_pos = Position::new(inst.target[0], inst.target[1]);
    let target = Target::Red(Symbol::Circle);
    let round = Round::new(board.clone(), target, target_pos);

    let t0 = Instant::now();

    // Catch panics from the solver (e.g., unsolvable boards)
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut solver = AStar::new();
        solver.solve(&round, start_positions.clone())
    }));

    let elapsed_us = t0.elapsed().as_micros();

    match result {
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "solver panicked".to_string()
            };
            Result {
                solved: false,
                moves: 0,
                path: vec![],
                time_us: elapsed_us,
                error: Some(msg),
            }
        }
        Ok(path) => {
            let num_moves = path.len();

            // Simulate path to get intermediate positions
            let mut cur_pos = start_positions;
            let mut path_moves = Vec::with_capacity(num_moves);

            for &(robot, direction) in path.movements() {
                let from_pos = cur_pos[robot];
                let new_pos = cur_pos.clone().move_in_direction(&board, robot, direction);
                let to_pos = new_pos[robot];
                cur_pos = new_pos;

                path_moves.push(Move {
                    robot: robot_name(robot),
                    direction: direction_name(direction),
                    from: [from_pos.column(), from_pos.row()],
                    to: [to_pos.column(), to_pos.row()],
                });
            }

            Result {
                solved: true,
                moves: num_moves,
                path: path_moves,
                time_us: elapsed_us,
                error: None,
            }
        }
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    let instances: Vec<Instance> = serde_json::from_str(&input).expect("Failed to parse JSON input");

    let results: Vec<Result> = instances.iter().map(solve_instance).collect();

    let output = serde_json::to_string(&results).expect("Failed to serialize results");
    println!("{}", output);
}
