use std::sync::Arc;

use cgmath::{Point3, Vector3, InnerSpace};

use ::geometric::{HasPosition};
use ::materials::Material;

pub struct Object {
    pub material: Arc<Material>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ObjectIndex(pub usize);

pub struct Sphere {
    pub object_index: ObjectIndex,
    pub geometry: SphereGeometry,
}

#[derive(Clone, Copy)]
pub struct SphereGeometry {
    pub center: Point3<f32>,
    pub radius: f32,
}

pub struct Triangle<T> {
    pub object_index: ObjectIndex,
    pub vertices: [T; 3],
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum PrimitiveIndex {
    Sphere(usize), Triangle(usize)
}

impl<T: HasPosition> Triangle<T> {
    pub fn face_normal(&self) -> Vector3<f32> {
        let a = self.vertices[1].get_position() - self.vertices[0].get_position();
        let b = self.vertices[2].get_position() - self.vertices[1].get_position();

        a.cross(b).normalize()
    }

    pub fn backface(&self, ray_dir: &Vector3<f32>) -> bool {
        self.face_normal().dot(*ray_dir) > 0.0
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct VertexIndex(pub usize);
