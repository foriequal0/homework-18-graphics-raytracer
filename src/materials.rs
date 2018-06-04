use std;

use cgmath::{EuclideanSpace, InnerSpace, Vector3};
use palette::{LinSrgb};

use ::consts;
use ::geometric::PositionNormalUV;

pub struct MaterialProbe
{
    pub at: PositionNormalUV,
    pub view_direction: Vector3<f32>,
    pub light_direction: Vector3<f32>,
}

pub trait Material {
    fn adjust_normal(&self, at: PositionNormalUV) -> Vector3<f32>;

    fn get_diffuse(&self, probe: &MaterialProbe) -> LinSrgb;
    fn get_specular(&self, probe: &MaterialProbe) -> LinSrgb;
    fn get_shiness(&self) -> f32;

    fn get_refraction(&self, probe: &MaterialProbe) -> LinSrgb {
        let shiness = self.get_shiness();
        let diffuse = self.get_diffuse(&probe);
        let specular = self.get_specular(&probe);

        diffuse * (1.0 - shiness) + specular * shiness
    }
}

pub struct ColorMaterial {
    pub diffuse_color: LinSrgb,
    pub shiness: f32,
    pub specular_color: LinSrgb,
    pub smoothness: f32,
}

impl Material for ColorMaterial {
    fn adjust_normal(&self, at: PositionNormalUV) -> Vector3<f32> {
        at.normal
    }

    fn get_diffuse(&self, probe: &MaterialProbe) -> LinSrgb {
        let cosine = probe.light_direction.dot(probe.at.normal);
        if cosine > 0.0 {
            self.diffuse_color * cosine
        } else {
            consts::linsrgb::black()
        }
    }

    fn get_specular(&self, probe: &MaterialProbe) -> LinSrgb {
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

    fn get_shiness(&self) -> f32 {
        self.shiness
    }
}
