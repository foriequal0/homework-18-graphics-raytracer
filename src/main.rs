extern crate cgmath;
#[macro_use]
extern crate itertools;
extern crate num_traits;
#[macro_use]
extern crate palette;
extern crate png;

use cgmath::{Angle, Deg, EuclideanSpace, InnerSpace, Point3, Vector2, Vector3};
use image::Image;
use palette::{LinSrgb, Srgb};
use photon::PhotonAccumulator;
use png::{Encoder, HasParameters};
use std::convert::{From, Into};
use std::fs::File;
use std::rc::Rc;

mod photon;
mod image;
mod consts;

struct Camera {
    fovy: cgmath::Rad<f32>,
    center: Point3<f32>,
    toward: Vector3<f32>,
    up: Vector3<f32>,
    near: f32,
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
        }
    }
}

struct Ray {
    origin: Point3<f32>,
    direction: Vector3<f32>,
}

impl Camera {
    fn shoot_into(screen_position: Point3<f32>) -> Ray {
        Ray {
            origin: (0.0, 0.0, 0.0).into(),
            direction: (0.0, 0.0, 0.0).into(),
        }
    }
}

trait Material {
    fn get_diffuse(&self) -> LinSrgb;
    fn get_specular(&self) -> LinSrgb;

    fn get_normal(&self) -> Vector3<f32>;
}

struct ColorMaterial {
    diffuse: LinSrgb,
    specular: LinSrgb,
}

impl Material for ColorMaterial {
    fn get_diffuse(&self) -> LinSrgb {
        self.diffuse
    }

    fn get_specular(&self) -> LinSrgb {
        self.specular
    }

    fn get_normal(&self) -> Vector3<f32> {
        -Vector3::unit_z()
    }
}

struct Object {
    material: Rc<Material>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct ObjectIndex(usize);

struct Sphere {
    object_index: ObjectIndex,
    geometry: SphereGeometry,
}

#[derive(Clone, Copy)]
struct SphereGeometry {
    center: cgmath::Point3<f32>,
    radius: f32,
}

struct Triangle<T> {
    object_index: ObjectIndex,
    vertices: [T; 3],
}

impl<T: HasPosition> Triangle<T> {
    fn face_normal(&self) -> Vector3<f32> {
        let a = self.vertices[1].get_position() - self.vertices[0].get_position();
        let b = self.vertices[2].get_position() - self.vertices[1].get_position();

        a.cross(b).normalize()
    }

    fn backface(&self, ray_dir: &Vector3<f32>) -> bool {
        self.face_normal().dot(*ray_dir) > 0.0
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct VertexIndex(usize);

trait HasPosition {
    fn get_position(&self) -> Point3<f32>;
}

#[derive(Clone, Copy)]
struct Position {
    position: Point3<f32>,
}

impl HasPosition for Position {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
struct PositionNormal {
    position: Point3<f32>,
    normal: Vector3<f32>,
}

impl HasPosition for PositionNormal {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
struct PositionUV {
    position: Point3<f32>,
    uv: Point3<f32>,
}

impl HasPosition for PositionUV {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

#[derive(Clone, Copy)]
struct PositionNormalUV {
    position: Point3<f32>,
    normal: Vector3<f32>,
    uv: Point3<f32>,
}

impl HasPosition for PositionNormalUV {
    fn get_position(&self) -> Point3<f32> {
        self.position
    }
}

struct Directional {
    direction: Vector3<f32>,
    color: LinSrgb,
}

struct Spot {
    direction: Vector3<f32>,
    steradian: f32,
    color: LinSrgb,
}

struct Point {
    direction: Vector3<f32>,
    color: LinSrgb,
}

enum Light {
    Directional(Directional),
    Spot(Spot),
    Point(Point),
}

impl std::convert::From<Directional> for Light {
    fn from(x: Directional) -> Light { Light::Directional(x) }
}

impl std::convert::From<Spot> for Light {
    fn from(x: Spot) -> Light { Light::Spot(x) }
}

impl std::convert::From<Point> for Light {
    fn from(x: Point) -> Light { Light::Point(x) }
}

#[derive(Default)]
struct World {
    objects: Vec<Object>,

    triangles: Vec<Triangle<PositionNormalUV>>,
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
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

    fn push_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    fn cast(&self, ray: Ray, front: bool) -> f32 {
        for triangle in &self.triangles {
            let backface = triangle.backface(&ray.direction);
            if front && backface || !front && !backface {
                continue;
            }
            let n = triangle.face_normal();
            let d = n.dot(triangle.vertices[0].position.to_vec());
            let t = d - n.dot(ray.origin.to_vec()) / n.dot(ray.direction);
            if t < 0.0 {
                continue;
            }

            let q = ray.origin + ray.direction * t;

            let v = [
                triangle.vertices[0].position,
                triangle.vertices[1].position,
                triangle.vertices[2].position,
            ];

            let a = (v[1] - v[0]).cross(q - v[0]).dot(n);
            let b = (v[2] - v[1]).cross(q - v[1]).dot(n);
            let c = (v[0] - v[2]).cross(q - v[2]).dot(n);

            if a < 0.0 || b < 0.0 || c < 0.0 {
                continue;
            }
            return t;
        }
        std::f32::MAX
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

fn main() {
    let mut img = Image::<PhotonAccumulator>::new(320, 240);

    let mut world = World::new();
    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse: (1.0, 0.5, 0.2).into(),
                specular: consts::linsrgb::white(),
            })
        })
        .push_triangles(&square(&[
            PositionUV { position: (-0.5, 0.0, -0.5).into(), uv: (0.0, 0.0, 0.0).into() },
            PositionUV { position: (-0.5, 0.0, 0.5).into(), uv: (0.0, 0.0, 0.0).into() },
            PositionUV { position: (0.5, 0.0, 0.5).into(), uv: (0.0, 0.0, 0.0).into() },
            PositionUV { position: (0.5, 0.0, -0.5).into(), uv: (0.0, 0.0, 0.0).into() }
        ]));

    world.push_light(Directional {
        direction: Vector3::from((1.0, -1.0, 1.0)).normalize(),
        color: (1.0, 0.98, 0.95).into(),
    }.into());

    let camera = Camera {
        fovy: Deg(60.0).into(),
        center: (5.0, 5.0, 5.0).into(),
        toward: Vector3::from((-1.0, -1.0, -1.0)).normalize(),
        up: Vector3::from((0.0, 1.0, 0.0)).normalize(),
        near: -0.1,
    };

    for (y, x) in iproduct!(0..img.height, 0..img.width) {
        let clip_y = (img.height as f32 / 2.0 - y as f32) / img.height as f32;
        let clip_x = (x as f32 - img.width as f32 / 2.0) / img.height as f32;

        let ray = camera.shoot(&(clip_x, clip_y).into());
        let hit = world.cast(ray, true);
        if hit != std::f32::MAX {
            img[(x, y)].accumulate(consts::linsrgb::white())
        } else {
            img[(x, y)].accumulate(consts::linsrgb::black())
        }
    }

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
