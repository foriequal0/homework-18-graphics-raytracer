use std;
use std::convert::{From};
use cgmath::{Angle, Rad,
             Point3, Vector3, InnerSpace};
use palette::{LinSrgb};

#[derive(Clone)]
pub struct Directional {
    pub direction: Vector3<f32>,
    pub color: LinSrgb,
}

pub struct Spot {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub angle: Rad<f32>,
    pub softness: f32,
    pub color: LinSrgb,
}

pub struct Point {
    pub origin: Point3<f32>,
    pub color: LinSrgb,
}

pub enum Light {
    Directional(Directional),
    Spot(Spot),
    Point(Point),
}

impl From<Directional> for Light {
    fn from(x: Directional) -> Light { Light::Directional(x) }
}

impl From<Spot> for Light {
    fn from(x: Spot) -> Light { Light::Spot(x) }
}

impl From<Point> for Light {
    fn from(x: Point) -> Light { Light::Point(x) }
}

pub trait ApproximateIntoDirectional {
    fn approximate_into_directional(&self, position: Point3<f32>) -> Option<Directional>;
}

impl ApproximateIntoDirectional for Directional {
    fn approximate_into_directional(&self, position: Point3<f32>) -> Option<Directional> {
        Option::Some(self.clone())
    }
}

impl ApproximateIntoDirectional for Spot {
    fn approximate_into_directional(&self, position: Point3<f32>) -> Option<Directional> {
        let offset = position - self.origin;
        let angle = self.direction.angle(offset).0.abs();
        let spot_spread = self.angle.0;
        if angle > spot_spread {
            return Option::None
        }

        let angular_attenuation = (1.0 - angle / spot_spread).powf(self.softness + std::f32::EPSILON);
        let distance_attenuation = 1.0 / (offset.magnitude() + std::f32::EPSILON);
        return Option::Some(Directional {
            direction: (position - self.origin).normalize(),
            color: self.color * angular_attenuation * distance_attenuation,
        });
    }
}

impl ApproximateIntoDirectional for Point {
    fn approximate_into_directional(&self, position: Point3<f32>) -> Option<Directional> {
        let offset = position - self.origin;
        let distance_attenuation = 1.0 / (offset.magnitude() + std::f32::EPSILON);
        Option::Some(Directional {
            direction: offset.normalize(),
            color: self.color * distance_attenuation,
        })
    }
}

impl ApproximateIntoDirectional for Light {
    fn approximate_into_directional(&self, position: Point3<f32>) -> Option<Directional> {
        match self {
            Light::Directional(d) => d.approximate_into_directional(position),
            Light::Spot(s) => s.approximate_into_directional(position),
            Light::Point(p) => p.approximate_into_directional(position),
        }
    }
}
