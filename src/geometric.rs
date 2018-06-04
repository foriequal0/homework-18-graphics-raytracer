use cgmath::{Point3, Point2, Vector3};

pub trait HasPosition {
    fn get_position(&self) -> Point3<f32>;
}

#[derive(Clone, Copy)]
pub struct Position {
    pub position: Point3<f32>,
}

impl HasPosition for Position {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
pub struct PositionNormal {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
}

impl HasPosition for PositionNormal {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
pub struct PositionUV {
    pub position: Point3<f32>,
    pub uv: Point2<f32>,
}

impl HasPosition for PositionUV {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
pub struct PositionNormalUV {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub uv: Point2<f32>,
}

impl HasPosition for PositionNormalUV {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}
