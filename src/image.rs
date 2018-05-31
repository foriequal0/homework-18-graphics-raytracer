use std::vec::Vec;
use std::iter;
use std::ops::{Index, IndexMut};

use palette::{IntoColor, Component, Pixel};
use palette::rgb::{Rgb, RgbStandard, RgbSpace};

use ::photon::PhotonAccumulator;

pub struct Image<P> {
    pub width: usize,
    pub height: usize,
    pixels: Vec<P>,
}

impl<P> Image<P>
where
    P: Default + Copy
{
    pub fn new(width: usize, height: usize) -> Image<P> {
        let pixels: Vec<P> = iter::repeat(Default::default())
            .take(width * height)
            .collect();

        Image { width, height, pixels }
    }

    pub fn as_slice(&self) -> &[P] {
        &self.pixels
    }

    pub fn as_slice_mut(&mut self) -> &mut[P] {
        &mut self.pixels
    }
}

impl<P> Index<(usize, usize)> for Image<P> {
    type Output = P;
    fn index(&self, index: (usize, usize)) -> &P {
        let (x, y) = index;
        &self.pixels[y * self.width + x]
    }
}

impl<P> IndexMut<(usize, usize)> for Image<P> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut P {
        let (x, y) = index;
        &mut self.pixels[y * self.width + x]
    }
}

impl<S, T> Image<Rgb<S, T>>
where
    S: RgbStandard,
    T: Component,
{
    pub fn convert_from<Q>(img: &Image<Q>) -> Image<Rgb<S, T>>
    where
        Q: Clone + IntoColor<<S::Space as RgbSpace>::WhitePoint, f32>
    {
        Image {
            width: img.width,
            height: img.height,
            pixels: img.pixels.iter().cloned()
                .map(|x| x.into_rgb().into_encoding().into_format())
                .collect()
        }
    }

    pub fn as_raw_slice(&self) -> &[T] {
        Pixel::into_raw_slice(&self.pixels)
    }

    pub fn as_raw_slice_mut(&mut self) -> &mut [T] {
        Pixel::into_raw_slice_mut(&mut self.pixels)
    }
}