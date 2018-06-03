extern crate cgmath;
#[macro_use]
extern crate itertools;
extern crate num_traits;
#[macro_use]
extern crate palette;
extern crate png;
extern crate stopwatch;

use cgmath::{Angle, Rad, Deg,
             EuclideanSpace, InnerSpace,
             Point2, Point3, Vector2, Vector3, Matrix3};
use image::Image;
use palette::{LinSrgb, Srgb};
use photon::PhotonAccumulator;
use png::{Encoder, HasParameters};
use std::convert::{From, Into};
use std::fs::File;
use std::rc::Rc;
use std::borrow::Borrow;

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


#[derive(PartialEq, Eq, Clone, Copy)]
enum FaceDirection {
    Front,
    Back,
    Both
}

struct Ray {
    origin: Point3<f32>,
    direction: Vector3<f32>,
    exclude: Option<PrimitiveIndex>,
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

trait Material {
    fn get_diffuse(&self) -> LinSrgb;
    fn get_specular(&self) -> Specular;

    fn get_normal(&self) -> Vector3<f32>;
}

struct ColorMaterial {
    diffuse: LinSrgb,
    specular: Specular,
}

#[derive(Clone, Copy)]
struct Specular {
    color: LinSrgb,
    shiness: f32,
}

impl Material for ColorMaterial {
    fn get_diffuse(&self) -> LinSrgb {
        self.diffuse
    }

    fn get_specular(&self) -> Specular {
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

#[derive(PartialEq, Eq, Clone, Copy)]
enum PrimitiveIndex {
    Sphere(usize), Triangle(usize)
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
    uv: Point2<f32>,
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
    uv: Point2<f32>,
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
    origin: Point3<f32>,
    direction: Vector3<f32>,
    angle: Rad<f32>,
    softness: f32,
    color: LinSrgb,
}

struct Point {
    origin: Point3<f32>,
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

struct Hit<'a> {
    object: &'a Object,
    index: PrimitiveIndex,
    at: PositionNormalUV,
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
        let mut some_nearest_t = Option::None;
        let mut some_nearest_result = Option::None;
        for (i, triangle) in self.triangles.iter().enumerate() {
            if ray.exclude == PrimitiveIndex::Triangle(i).into() {
                // cast가 시작한 객체랑 같음
                continue;
            }

            let backface = triangle.backface(&ray.direction);
            if (ray.face_direction == FaceDirection::Front && backface)
                || (ray.face_direction == FaceDirection::Back && !backface)
            {
                // 뒤집어짐
                continue;
            }
            let face_normal = triangle.face_normal();
            let d = face_normal.dot(triangle.vertices[0].position.to_vec());
            let t = (d - face_normal.dot(ray.origin.to_vec())) / face_normal.dot(ray.direction);
            if t <= 0.0 {
                // 광선 진행방향 뒷편에 존재함
                continue;
            }

            let position = ray.origin + ray.direction * t;

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

            if let Option::Some(nearest_t) = some_nearest_t {
                if nearest_t < t {
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

            let normal = normals * barycentric;
            let uv = Point2::from_vec(uvs[0] * barycentric[0] + uvs[1] * barycentric[1] + uvs[2] * barycentric[2]);
            some_nearest_t = t.into();
            some_nearest_result = Hit {
                object: &self.objects[triangle.object_index.0],
                index: PrimitiveIndex::Triangle(i),
                at: PositionNormalUV { position, normal, uv }
            }.into();
        }

        for (i, sphere) in self.spheres.iter().enumerate() {
            if ray.exclude == PrimitiveIndex::Sphere(i).into() {
                // cast가 시작한 객체랑 같음
                continue;
            }

            let distance = (sphere.geometry.center - ray.origin).cross(ray.direction).magnitude();
            if distance > sphere.geometry.radius {
                continue
            }

            let displacement = sphere.geometry.center - ray.origin;
            let tc = ray.direction.dot(displacement);
            if tc < 0.0 {
                continue;
            }

            let k = (sphere.geometry.radius.powi(2) - distance.powi(2)).sqrt();
            let t = match ray.face_direction{
                FaceDirection::Front => tc - k,
                FaceDirection::Back => tc + k,
                FaceDirection::Both => if tc > k { tc - k } else {tc + k}
            };

            if let Option::Some(nearest_t) = some_nearest_t {
                if nearest_t < t {
                    continue;
                }
            }

            let position = ray.origin + ray.direction * t;
            let normal = (position - sphere.geometry.center).normalize();
            let uv = Point2 {
                x: normal.y.acos() / std::f32::consts::PI,
                y: normal.z.atan2(normal.x) / (std::f32::consts::PI * 2.0) + 0.5,
            };

            some_nearest_t = t.into();
            some_nearest_result = Hit {
                object: &self.objects[sphere.object_index.0],
                index: PrimitiveIndex::Sphere(i),
                at: PositionNormalUV { position, normal, uv }
            }.into();
        }
        some_nearest_result
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
    let mut img = Image::<LinSrgb>::new(640, 480);

    let mut world = World::new();
    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse: (1.0, 0.5, 0.2).into(),
                specular: Specular {
                    color: consts::linsrgb::white(),
                    shiness: 10.0
                }
            })
        })
        .push_triangles(&square(&[
            PositionUV { position: (-0.5, 0.0, -0.5).into(), uv: (0.0, 0.0).into() },
            PositionUV { position: (-0.5, 0.0, 0.5).into(), uv: (0.0, 1.0).into() },
            PositionUV { position: (0.5, 0.0, 0.5).into(), uv: (1.0, 0.0).into() },
            PositionUV { position: (0.5, 0.0, -0.5).into(), uv: (0.0, 1.0).into() }
        ]));

    world
        .push_object(Object {
            material: Rc::new(ColorMaterial {
                diffuse: (0.5, 1.0, 0.2).into(),
                specular: Specular {
                    color: consts::linsrgb::white()/3.0,
                    shiness: 10.0
                }
            })
        })
        .push_sphere(&SphereGeometry{
            center: (0.0, 0.5, 0.0).into(),
            radius: 0.5,
        });

    world.push_light(Directional {
        direction: Vector3::new(-1.0, -1.0, 0.0).normalize(),
        color: (1.0, 0.98, 0.95).into(),
    });

    world.push_light(Spot {
        origin: Point3::new(-0.5, 2.0, 0.5),
        direction: Vector3::new(0.0, -1.0, 0.0),
        angle: Deg(20.0).into(),
        softness: 0.5,
        color: LinSrgb::new(1.0, 0.0, 0.0)
    });

    world.push_light(Point {
        origin: Point3::new(-0.5, 0.1, 0.5),
        color: LinSrgb::new(0.0, 0.0, 1.0)
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
        let hit = world.cast(&ray);

        if hit.is_none() {
            continue;
        }
        let hit = hit.unwrap();
        let material = hit.object.material.borrow();
        let normal = hit.at.normal;

        let get_color = |to_view: Vector3<f32>, material: &Material, light: &Directional| {
            let to_light = -light.direction;
            let shadow_ray = Ray {
                origin: hit.at.position,
                direction: to_light,
                exclude: hit.index.into(),
                face_direction: FaceDirection::Both,
            };
            let in_shadow = world.cast(&shadow_ray).is_some();
            if in_shadow {
                return Option::None;
            }
            let correlation = to_light.dot(normal);
            if correlation <= 0.0 {
                return Option::None;
            }
            let diffuse_color = material.get_diffuse() * light.color * correlation;

            let reflected_ray = 2.0 * correlation * normal - to_light;
            let specular = material.get_specular();
            let specular_amount = reflected_ray.dot(to_view).max(0.0);
            let specular_color = (specular.color * light.color) * (specular_amount.powf(specular.shiness));
            Option::Some(diffuse_color + specular_color)
        };
        for light in &world.lights {
            let some_color = match light {
                Light::Directional(light) => get_color(-ray.direction, material, light),
                Light::Spot(spot) => {
                    let offset = hit.at.position - spot.origin;
                    let angle = spot.direction.angle(offset).0.abs();
                    let spot_spread = spot.angle.0;
                    if angle > spot_spread {
                        Option::None
                    } else {
                        let angular_attenuation = (1.0 - angle / spot_spread).powf(spot.softness + std::f32::EPSILON);
                        let distance_attenuation = 1.0 / (offset.magnitude() + std::f32::EPSILON);
                        get_color(-ray.direction, material, &Directional {
                            direction: (hit.at.position - spot.origin).normalize(),
                            color: spot.color * angular_attenuation * distance_attenuation,
                        })
                    }
                },
                Light::Point(point) => {
                    let offset = hit.at.position - point.origin;
                    let distance_attenuation = 1.0 / (offset.magnitude() + std::f32::EPSILON);
                    get_color(-ray.direction, material, &Directional {
                        direction: offset.normalize(),
                        color: point.color * distance_attenuation,
                    })
                }
                _ => {
                    unreachable!();
                }
            };
            if let Option::Some(color) = some_color {
                img[(x, y)] = img[(x, y)] + color;
            }
        }

        if i % 50000 == 0 {
            let i = i as i64;
            let elapsed = sw.elapsed_ms();
            println!("{} rays in {} ms (avg: {} ray/s)", i, elapsed, i/(elapsed+1) * 1000);
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
