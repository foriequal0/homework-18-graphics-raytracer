extern crate cgmath;
#[macro_use]
extern crate itertools;
extern crate num_traits;
#[macro_use]
extern crate palette;
extern crate png;
extern crate stopwatch;

mod photon;
mod image;
mod lights;
mod materials;
mod geometric;
mod primitives;
mod consts;

use std::convert::{From, Into};
use std::fs::File;
use std::rc::Rc;
use std::borrow::Borrow;

use cgmath::{Angle, Deg,
             EuclideanSpace, InnerSpace, MetricSpace,
             Point2, Point3, Vector2, Vector3, Matrix3};
use palette::{LinSrgb, Srgb, IntoColor, Mix};
use png::{Encoder, HasParameters};

use lights::{Light, Directional, Spot, Point, ApproximateIntoDirectional };
use materials::{MaterialProbe, Material, ColorMaterial};
use primitives::{PrimitiveIndex, Object, Triangle, Sphere, ObjectIndex, SphereGeometry};
use geometric::{PositionNormalUV, PositionUV};
use image::Image;

struct Camera {
    fovy: cgmath::Rad<f32>,
    center: Point3<f32>,
    toward: Vector3<f32>,
    up: Vector3<f32>,
    near: f32,
}


#[derive(PartialEq, Eq, Clone, Copy)]
enum FaceDirection {
    Front,
    Back,
    Both
}

impl FaceDirection {
    fn invert(self) -> FaceDirection {
        match self {
            FaceDirection::Front => FaceDirection::Back,
            FaceDirection::Back => FaceDirection::Front,
            FaceDirection::Both => FaceDirection::Both,
        }
    }
}

#[derive(Clone, Copy)]
struct Ray {
    origin: Point3<f32>,
    direction: Vector3<f32>,
    face_direction: FaceDirection,
    exclude: Option<Exclusion>,
}

#[derive(Clone, Copy)]
struct Exclusion {
    index: PrimitiveIndex,
    face_direction: FaceDirection,
}

impl Camera {
    fn shoot(&self, clip: &Vector2<f32>) -> Ray {
        let toward = self.toward.normalize();
        let right = toward.cross(self.up).normalize();
        let up = right.cross(toward).normalize();

        let x = (self.fovy / 2.0).tan() * right;
        let y = (self.fovy / 2.0).tan() * up;
        let direction = (clip.x * x + clip.y * y + toward).normalize();
        let origin = self.center + toward * self.near;
        Ray {
            origin,
            direction,
            exclude: Option::None,
            face_direction: FaceDirection::Front,
        }
    }
}

#[derive(Default)]
struct World {
    objects: Vec<Object>,

    triangles: Vec<Triangle<PositionNormalUV>>,
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
}

struct Hit<'a> {
    object: &'a Object,
    ray: Ray,
    index: PrimitiveIndex,
    at: PositionNormalUV,
    face_direction: FaceDirection,
    distance: f32,
}

enum Refraction {
    Escaped {
        travel_distance: f32,
        escape_ray: Ray,
    },
    Infinite {
        ray: Ray,
    },
    Trapped
}

impl World {
    fn new() -> World {
        World {
            ..Default::default()
        }
    }

    fn push_object<'a>(&'a mut self, object: Object) -> ObjectProxy<'a> {
        self.objects.push(object);
        let object_index = ObjectIndex(self.objects.len() - 1);
        ObjectProxy {
            object_index: object_index,
            world: self,
        }
    }

    fn push_light<L:Into<Light>>(&mut self, light: L) {
        self.lights.push(light.into());
    }

    fn cast(&self, ray: &Ray) -> Option<Hit> {
        let mut nearest_distance = Option::None;
        let mut nearest_result = Option::None;
        for (i, triangle) in self.triangles.iter().enumerate() {
            let backface = triangle.backface(&ray.direction);
            if backface && ray.face_direction == FaceDirection::Front
                || !backface && ray.face_direction == FaceDirection::Back {
                continue;
            }

            if let Option::Some(exclusion) = ray.exclude {
                let same_face = exclusion.index == PrimitiveIndex::Triangle(i);
                let criteria = match exclusion.face_direction {
                    FaceDirection::Front => !backface,
                    FaceDirection::Back => backface,
                    FaceDirection::Both => true
                };
                if same_face && criteria {
                    continue;
                }
            }

            let face_normal = triangle.face_normal();
            let d = face_normal.dot(triangle.vertices[0].position.to_vec());
            let travel_distance = (d - face_normal.dot(ray.origin.to_vec())) / face_normal.dot(ray.direction);
            if travel_distance <= 0.0 {
                // 광선 진행방향 뒷편에 존재함
                continue;
            }

            let position = ray.origin + ray.direction * travel_distance;

            let v = [
                triangle.vertices[0].position,
                triangle.vertices[1].position,
                triangle.vertices[2].position,
            ];

            let area = [
                (v[2] - v[1]).cross(position - v[1]).dot(face_normal),
                (v[0] - v[2]).cross(position - v[2]).dot(face_normal),
                (v[1] - v[0]).cross(position - v[0]).dot(face_normal),
            ];

            if area.iter().any(|x| *x < 0.0) {
                // 삼각형 밖에 존재함
                continue;
            }

            if let Option::Some(nearest_t) = nearest_distance {
                if nearest_t < travel_distance {
                    continue;
                }
            }

            let area_of_triangle = (v[1] - v[0]).cross(v[2] - v[0]).dot(face_normal);
            let barycentric = Vector3::from(area) / area_of_triangle;
            let normals = Matrix3::from_cols(
                triangle.vertices[0].normal,
                triangle.vertices[1].normal,
                triangle.vertices[2].normal
            );
            let uvs = [
                triangle.vertices[0].uv.to_vec(),
                triangle.vertices[1].uv.to_vec(),
                triangle.vertices[2].uv.to_vec()
            ];

            let normal = {
                let mut tmp = normals * barycentric;
                if backface { -tmp } else { tmp }
            };
            let uv = Point2::from_vec(uvs[0] * barycentric[0] + uvs[1] * barycentric[1] + uvs[2] * barycentric[2]);
            nearest_distance = travel_distance.into();
            nearest_result = Hit {
                object: &self.objects[triangle.object_index.0],
                ray: *ray,
                index: PrimitiveIndex::Triangle(i),
                at: PositionNormalUV { position, normal, uv },
                distance: travel_distance,
                face_direction: if backface { FaceDirection::Back } else { FaceDirection::Front }
            }.into();
        }

        for (i, sphere) in self.spheres.iter().enumerate() {
            let line_sphere_distance = (sphere.geometry.center - ray.origin).cross(ray.direction).magnitude();
            if line_sphere_distance > sphere.geometry.radius {
                continue
            }

            let displacement = sphere.geometry.center - ray.origin;
            let tc = ray.direction.dot(displacement);
            let k = (sphere.geometry.radius.powi(2) - line_sphere_distance.powi(2)).sqrt();
            let (travel_distance, backface) = match ray.face_direction{
                FaceDirection::Front => (tc - k, false),
                FaceDirection::Back => (tc + k, true),
                FaceDirection::Both => if tc < k {
                    (tc + k, true)
                } else {
                    (tc - k, false)
                }
            };
            if travel_distance <= 0.0 {
                continue;
            }

            if let Option::Some(exclusion) = ray.exclude {
                let same_face = exclusion.index == PrimitiveIndex::Sphere(i);
                let criteria = match exclusion.face_direction {
                    FaceDirection::Front => !backface,
                    FaceDirection::Back => backface,
                    FaceDirection::Both => true
                };
                if same_face && criteria {
                    continue;
                }
            }

            if let Option::Some(nearest_t) = nearest_distance {
                if nearest_t < travel_distance {
                    continue;
                }
            }

            let position = ray.origin + ray.direction * travel_distance;
            let normal = {
                let tmp = (position - sphere.geometry.center).normalize();
                if backface { -tmp } else { tmp }
            };

            let uv = Point2 {
                x: normal.y.acos() / std::f32::consts::PI,
                y: normal.z.atan2(normal.x) / (std::f32::consts::PI * 2.0) + 0.5,
            };

            nearest_distance = travel_distance.into();
            nearest_result = Hit {
                object: &self.objects[sphere.object_index.0],
                ray: *ray,
                index: PrimitiveIndex::Sphere(i),
                at: PositionNormalUV { position, normal, uv },
                distance: travel_distance,
                face_direction: if backface { FaceDirection::Back } else { FaceDirection::Front }
            }.into();
        }
        nearest_result
    }

    fn get_reflect(&self, hit: &Hit) -> Ray {
        let reflect = |n: Vector3<f32>, l: Vector3<f32>| l - 2.0 * l.dot(n) * n;
        let reflected = reflect(hit.at.normal, hit.ray.direction);
        let ray = Ray {
            origin: hit.at.position,
            direction: reflected.normalize(),
            face_direction: hit.ray.face_direction,
            exclude: Exclusion {
                index: hit.index,
                face_direction: hit.face_direction.invert(),
            }.into(),
        };
        ray
    }

    fn get_refract<'a>(&self, hit: &'a Hit, max_distance: f32) -> Refraction {
        let refract = |n: Vector3<f32>, l: Vector3<f32>, k: f32| {
            let cos = -l.dot(n);
            if k.powi(2) >= 1.0 - cos.powi(2) {
                Option::Some((l + n * cos) / k - n * (1.0 - (1.0 - cos.powi(2)) / k.powi(2)).sqrt())
                    .map(|x| x.normalize())
            } else {
                Option::None
            }
        };

        let k = hit.object.material.get_refraction_index();
        let refract_in = refract(hit.at.normal, hit.ray.direction, k)
            .expect("Refraction index must larger than 1.0, then it'll never fail");
        let ray_inside = Ray {
            origin: hit.at.position,
            direction: refract_in.normalize(),
            face_direction: FaceDirection::Back,
            exclude: Exclusion {
                index: hit.index,
                face_direction: FaceDirection::Front,
            }.into()
        };

        if hit.index == PrimitiveIndex::Sphere(1) {
            let x = hit.ray.direction.normalize();
            let y = refract_in.normalize();
            let z = 1.0;
        }
        // get out
        let mut hit_inside = match self.cast(&ray_inside) {
            Option::Some(hit) => hit,
            Option::None => { return Refraction::Infinite { ray: ray_inside }; }
        };
        let mut travel_distance = hit_inside.at.position.distance(hit.at.position);
        let mut refract_out = refract(hit_inside.at.normal, hit_inside.ray.direction, 1.0/k);
        let mut retry = 0;
        while refract_out.is_none() && travel_distance <= max_distance && retry < 10 {
            let previous_hit_position = hit_inside.at.position;
            let total_reflect = self.get_reflect(&hit_inside);
            hit_inside = match self.cast(&total_reflect) {
                Option::Some(hit) => hit,
                Option::None => { return Refraction::Infinite { ray: total_reflect }; }
            };
            travel_distance += previous_hit_position.distance(hit_inside.at.position);
            refract_out = refract(hit_inside.at.normal, hit_inside.ray.direction, 1.0/k);
            retry += 1;
        }

        match refract_out {
            Option::None => Refraction::Trapped,
            Option::Some(out) => {
                let escape_ray = Ray {
                    origin: hit_inside.at.position,
                    direction: out.normalize(),
                    face_direction: FaceDirection::Front,
                    exclude: Exclusion {
                        index: hit_inside.index,
                        face_direction: FaceDirection::Back,
                    }.into()
                };
                Refraction::Escaped { travel_distance, escape_ray }
            }
        }
    }

    fn get_shade(&self, hit: &Hit) -> LinSrgb {
        let material: &Material = hit.object.material.borrow();
        let ray = hit.ray;
        let normal = material.adjust_normal(hit.at);

        let mut sum = LinSrgb::new(0.0, 0.0, 0.0);
        for light in &self.lights {
            let approx_directional = light.approximate_into_directional(hit.at.position);
            if approx_directional.is_none() {
                continue;
            }
            let light = approx_directional.unwrap();

            let cosine = -light.direction.dot(normal);
            if cosine <= 0.0 {
                continue;
            }

            // TODO: shadow ray through transparent object
            let shadow_ray = Ray {
                origin: hit.at.position,
                direction: -light.direction,
                face_direction: FaceDirection::Both,
                exclude: Exclusion {
                    index: hit.index,
                    face_direction: FaceDirection::Both,
                }.into()
            };

            if let Option::Some(occlusion) = self.cast(&shadow_ray) {
                match light.origin {
                    Option::Some(light_origin) => {
                        let occlusion_distance = hit.at.position.distance(occlusion.at.position);
                        let light_distance = hit.at.position.distance(light_origin);
                        if occlusion_distance < light_distance {
                            continue;
                        }
                    },
                    Option::None => {
                        continue;
                    }
                }
            }

            let probe = MaterialProbe {
                at: PositionNormalUV { position: hit.at.position, normal: normal, uv: hit.at.uv },
                view_direction: -ray.direction,
                light_direction: -light.direction,
            };

            // to_light, normal
            let shiness = material.get_shiness();
            let diffuse = material.get_diffuse(&probe) * light.color;
            let specular = material.get_specular(&probe) * light.color;
            sum = sum + diffuse * (1.0 - shiness) + specular * shiness;
        }
        sum
    }
}

struct ObjectProxy<'a> {
    object_index: ObjectIndex,
    world: &'a mut World,
}

impl<'a> ObjectProxy<'a> {
    fn push_triangle(&mut self, vertices: &[PositionNormalUV; 3]) -> &mut Self {
        self.world.triangles.push(Triangle {
            object_index: self.object_index,
            vertices: *vertices,
        });
        self
    }

    fn push_sphere(&mut self, geometry: &SphereGeometry) -> &mut Self {
        self.world.spheres.push(Sphere {
            object_index: self.object_index,
            geometry: *geometry,
        });
        self
    }

    fn push_triangles(&mut self, triangles: &[[PositionNormalUV; 3]]) -> &mut Self {
        for vertices in triangles {
            self.push_triangle(vertices);
        }
        self
    }
}

fn triangle(vertices: &[PositionUV; 3]) -> [PositionNormalUV; 3] {
    let a: Vector3<f32> = vertices[1].position - vertices[0].position;
    let b: Vector3<f32> = vertices[2].position - vertices[1].position;
    let normal = a.cross(b).normalize();
    [
        PositionNormalUV { position: vertices[0].position, normal: normal, uv: vertices[0].uv },
        PositionNormalUV { position: vertices[1].position, normal: normal, uv: vertices[1].uv },
        PositionNormalUV { position: vertices[2].position, normal: normal, uv: vertices[2].uv },
    ]
}

fn square(vertices: &[PositionUV; 4]) -> [[PositionNormalUV; 3]; 2] {
    [
        triangle(&[vertices[0], vertices[1], vertices[2]]),
        triangle(&[vertices[0], vertices[2], vertices[3]])
    ]
}

fn post_process(img: &mut Image<LinSrgb>) {
    let mut luma_cumulative: Vec<f32> = img.as_slice().iter().cloned()
        .map(|x| x.into_luma().luma)
        .collect();
    luma_cumulative.sort_by(|a, b| (*a).partial_cmp(b).unwrap());
    let p98 = luma_cumulative[(luma_cumulative.len() as f32 * 0.99) as usize];
    if p98 > std::f32::EPSILON {
        for pixel in img.as_slice_mut() {
            *pixel = *pixel / p98;
        }
    }

    // TODO: highlight bloom effect
}

fn main() {
    let mut img = Image::<LinSrgb>::new(640, 480);

    let mut world = World::new();
    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse_color: (1.0, 0.8, 0.6).into(),
                shiness: 0.5,
                specular_color: consts::linsrgb::white(),
                smoothness: 1.0,
                refraction_index: 1.0,
                opaque_decay: 0.0,
                transparency: 0.0,
            })
        })
        .push_triangles(&square(&[
            PositionUV { position: (-2.0, 0.0, -2.0).into(), uv: (0.0, 0.0).into() },
            PositionUV { position: (-2.0, 0.0, 2.0).into(), uv: (0.0, 1.0).into() },
            PositionUV { position: (2.0, 0.0, 2.0).into(), uv: (1.0, 0.0).into() },
            PositionUV { position: (2.0, 0.0, -2.0).into(), uv: (0.0, 1.0).into() }
        ]));

    // 빨간공
    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse_color: (1.0, 0.2, 0.2).into(),
                shiness: 0.2,
                specular_color: consts::linsrgb::yellow(),
                smoothness: 0.2,
                refraction_index: 1.0,
                opaque_decay: 0.0,
                transparency: 0.0,
            })
        })
        .push_sphere(&SphereGeometry{
            center: (-0.5, 0.5, 0.5 / 3.0f32.sqrt()).into(),
            radius: 0.5,
        });

    world
        .push_object(Object{
            material: Rc::new(ColorMaterial {
                diffuse_color: (1.0, 1.0, 1.0).into(),
                shiness: 1.0,
                specular_color: consts::linsrgb::white(),
                smoothness: 0.01,
                refraction_index: 1.12,
                opaque_decay: 0.3,
                transparency: 0.96,
            })
        })
        .push_sphere(&SphereGeometry{
            center: (0.5, 0.5, 0.5 / 3.0f32.sqrt()).into(),
            radius: 0.5,
        });

    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse_color: (0.2, 0.5, 1.0).into(),
                shiness: 0.9,
                specular_color: consts::linsrgb::blue(),
                smoothness: 0.7,
                refraction_index: 1.0,
                opaque_decay: 0.0,
                transparency: 0.0,
            })
        })
        .push_sphere(&SphereGeometry{
            center: (0.0, 0.5, -1.0 / 3.0f32.sqrt()).into(),
            radius: 0.5,
        });

    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse_color: (0.5, 1.0, 0.2).into(),
                shiness: 0.5,
                specular_color: consts::linsrgb::white(),
                smoothness: 0.01,
                refraction_index: 1.0,
                opaque_decay: 0.0,
                transparency: 0.0,
            })
        })
        .push_sphere(&SphereGeometry{
            center: (0.0, 0.5 + (2.0f32/3.0).sqrt(), 0.0).into(),
            radius: 0.5,
        });

    world.push_light(Directional {
        origin: Option::None,
        direction: Vector3::new(-1.0, -1.0, 0.0).normalize(),
        color: LinSrgb::new(1.0, 0.98, 0.95),
    });

    world.push_light(Spot {
        origin: Point3::new(0.0, 10.0, 0.0),
        direction: Vector3::new(0.0, -1.0, -0.0).normalize(),
        angle: Deg(60.0).into(),
        softness: 1.0,
        color: LinSrgb::new(1.0, 0.5, 0.9) * 1.0f32
    });

    world.push_light(Point {
        origin: Point3::new(0.0, 0.1, 0.0),
        color: LinSrgb::new(0.8, 0.8, 1.0)
    });

    let camera = Camera {
        fovy: Deg(60.0).into(),
        center: (2.0, 2.5, 2.0).into(),
        toward: Vector3::new(-1.0, -1.0, -1.0).normalize(),
        up: Vector3::new(0.0, 1.0, 0.0).normalize(),
        near: -0.1,
    };

    let sw = stopwatch::Stopwatch::start_new();
    for (i, (y, x)) in iproduct!(0..img.height, 0..img.width).enumerate() {
        let clip_y = (img.height as f32 / 2.0 - y as f32) / img.height as f32;
        let clip_x = (x as f32 - img.width as f32 / 2.0) / img.height as f32;

        let ray = camera.shoot(&(clip_x, clip_y).into());
        let hit = match world.cast(&ray){
            Option::Some(hit) => hit,
            Option::None => { continue; }
        };

        let shade = world.get_shade(&hit);

        let shiness = hit.object.material.get_shiness();
        let reflection = if shiness > 0.0 {
            let reflected_ray = world.get_reflect(&hit);
            world.cast(&reflected_ray)
                .map(|reflected_hit| world.get_shade(&reflected_hit))
                .unwrap_or(LinSrgb::new(0.0, 0.0, 0.0))
        } else {
            LinSrgb::new(0.0, 0.0, 0.0)
        };

        let transparency = hit.object.material.get_transparency();
        let refraction = if transparency > 0.0 {
            match world.get_refract(&hit, 100.0) {
                Refraction::Escaped { travel_distance, escape_ray } => {
                    world.cast(&escape_ray)
                        .map(|escape_hit| {
                            let shade =  world.get_shade(&escape_hit);
                            let decay = hit.object.material.get_opaque_decay();
                            shade * decay.powf(travel_distance)
                        })
                        .unwrap_or(LinSrgb::new(0.0, 0.0, 0.0))
                },
                _ => LinSrgb::new(0.0, 0.0, 0.0)
            }
        } else {
            LinSrgb::new(0.0, 0.0, 0.0)
        };

        img[(x, y)] = img[(x, y)] + (shade + reflection * shiness).mix(&refraction, transparency);

        if i % 50000 == 0 {
            let i = i as i64;
            let elapsed = sw.elapsed_ms();
            println!("{} rays in {} ms (avg: {} ray/s)", i, elapsed, i/(elapsed+1) * 1000);
        }
    }
    post_process(&mut img);
    {
        let encoded = Image::<Srgb<u8>>::convert_from(&img);
        let out_file = File::create("./out.png").unwrap();
        let mut encoder = Encoder::new(
                out_file, encoded.width as u32, encoded.height as u32);
        encoder.set(png::ColorType::RGB);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(encoded.as_raw_slice()).unwrap();
    }
}
