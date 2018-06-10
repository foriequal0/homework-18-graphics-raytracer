use std;

use cgmath::{InnerSpace, Point2, Vector3};
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
        normal
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

    pub fn get_shade(&self, probe: &MaterialProbe) -> LinSrgb {
        let diffuse = self.get_diffuse(probe);
        let specular = self.get_specular(probe);

        diffuse.mix(&specular, self.shiness)
    }
}

#[derive(Clone)]
pub struct GenerativeMaterial<F>
where F: Fn(Point2<f32>) -> LinSrgb + Send + Sync
{
    pub diffuse_fn: F,
    pub shiness: f32,
    pub specular_color: LinSrgb,
    pub smoothness: f32,

    pub transparency: f32,
    pub refraction_index: f32,
    pub opaque_decay: f32,
}

impl<F> Material for GenerativeMaterial<F>
where F: Fn(Point2<f32>) -> LinSrgb + Send + Sync
{
    fn approx(&self, at: PositionNormalUV) -> ColorMaterial {
        let diffuse_fn = &self.diffuse_fn;
        ColorMaterial {
            diffuse_color: diffuse_fn(at.uv),
            shiness: self.shiness,
            specular_color: self.specular_color,
            smoothness: self.smoothness,
            transparency: self.transparency,
            refraction_index: self.refraction_index,
            opaque_decay: self.opaque_decay
        }
    }
}
