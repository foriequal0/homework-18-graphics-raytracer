extern crate png;
extern crate cgmath;
#[macro_use]
extern crate palette;
extern crate num_traits;

mod photon;
mod image;

use std::fs::File;

use palette::{LinSrgb, Srgb};
use png::{Encoder, HasParameters};

use photon::PhotonAccumulator;
use image::Image;

struct Camera {
    width: i32,
    height: i32,
    origin: cgmath::Point3<f32>,
    direction: cgmath::Vector3<f32>,
    perspective: cgmath::PerspectiveFov<f32>,
}

fn main() {
    let mut img = Image::<PhotonAccumulator>::new(320, 240);

    for y in 0..img.height {
        for x in 0..img.width {
            let lightness = x as f32 / img.width as f32;
            img[(x, y)].accumulate(LinSrgb::new(lightness, lightness, lightness));
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
    };
}
