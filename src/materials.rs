use std;

use cgmath::{InnerSpace, Point2, Vector3, Quaternion};
use palette::{LinSrgb, Mix};

use ::consts;
use ::geometric::PositionNormalUV;

pub struct MaterialProbe
{
    pub at: PositionNormalUV,
    pub view_direction: Vector3<f32>,
    pub light_direction: Vector3<f32>,
}

pub trait Material: Send + Sync {
    fn approx(&self, at: PositionNormalUV) -> ColorMaterial;
}

#[derive(Copy, Clone)]
pub struct ColorMaterial {
    pub normal: Vector3<f32>,
    pub diffuse_color: LinSrgb,
    pub shiness: f32,
    pub specular_color: LinSrgb,
    pub smoothness: f32,

    pub transparency: f32,
    pub refraction_index: f32,
    pub opaque_decay: f32,
}

impl Material for ColorMaterial {
    fn approx(&self, at: PositionNormalUV) -> ColorMaterial {
        self.clone()
    }
}

impl ColorMaterial{
    pub fn adjust_normal(&self, normal: Vector3<f32>) -> Vector3<f32> {
        let z = Vector3::new(0.0, 0.0, 1.0);
        let from_z = Quaternion::from_arc(z, normal, None);
        from_z * self.normal
    }

    pub fn get_diffuse(&self, probe: &MaterialProbe) -> LinSrgb {
        let cosine = probe.light_direction.dot(probe.at.normal);
        if cosine > 0.0 {
            self.diffuse_color * cosine
        } else {
            consts::linsrgb::black()
        }
    }

    pub fn get_specular(&self, probe: &MaterialProbe) -> LinSrgb {
        let cosine = probe.light_direction.dot(probe.at.normal);
        if cosine <= 0.0 {
            return consts::linsrgb::black()
        }
        let reflected_ray = 2.0 * cosine * probe.at.normal - probe.light_direction;
        let specular = 1.0/(self.smoothness + std::f32::EPSILON);
        let energy_conserving = (specular + 8.0) / (8.0 * std::f32::consts::PI);
        let specular_amount = reflected_ray.dot(probe.view_direction)
            .max(0.0).powf(specular) * energy_conserving;
        self.specular_color * specular_amount
    }
}

#[derive(Clone)]
pub struct GenerativeMaterial<F, G>
where F: Fn(Point2<f32>) -> LinSrgb + Send + Sync,
      G: Fn(Point2<f32>) -> Vector3<f32> + Send + Sync
{
    pub diffuse_fn: F,
    pub normal_fn: G,
    pub shiness: f32,
    pub specular_color: LinSrgb,
    pub smoothness: f32,

    pub transparency: f32,
    pub refraction_index: f32,
    pub opaque_decay: f32,
}

impl<F, G> Material for GenerativeMaterial<F, G>
where F: Fn(Point2<f32>) -> LinSrgb + Send + Sync,
      G: Fn(Point2<f32>) -> Vector3<f32> + Send + Sync
{
    fn approx(&self, at: PositionNormalUV) -> ColorMaterial {
        let diffuse_fn = &self.diffuse_fn;
        let normal_fn = &self.normal_fn;
        ColorMaterial {
            diffuse_color: diffuse_fn(at.uv),
            normal: normal_fn(at.uv),
            shiness: self.shiness,
            specular_color: self.specular_color,
            smoothness: self.smoothness,
            transparency: self.transparency,
            refraction_index: self.refraction_index,
            opaque_decay: self.opaque_decay
        }
    }
}
