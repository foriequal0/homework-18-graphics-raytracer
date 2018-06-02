use std;
use palette::{LinSrgb, Component, IntoColor};
use palette::rgb::{Rgb, RgbSpace};
use palette::encoding::Linear;
use palette::white_point::D65;

#[derive(Clone, Copy, IntoColor)]
#[palette_manual_into(Rgb = "into_rgb_internal")]
pub struct PhotonAccumulator {
    sum: LinSrgb,
    weight_sum: f32,
}

impl PhotonAccumulator {
    fn into_rgb_internal<S, T>(self) -> Rgb<Linear<S>, T>
    where
        S: RgbSpace<WhitePoint = D65>,
        T: Component
    {
        let avg = if self.weight_sum < std::f32::EPSILON { Rgb::new(0.0, 0.0, 0.0) }
            else { self.sum / self.weight_sum };
        avg.into_rgb().into_format()
    }

    pub fn accumulate(&mut self, photon: LinSrgb) {
        self.sum = self.sum + photon;
        self.weight_sum += 1.0;
    }

    pub fn accumulate_weight(&mut self, photon: LinSrgb, weight: f32) {
        self.sum = self.sum + photon * weight;
        self.weight_sum += weight;
    }
}

impl Default for PhotonAccumulator {
    fn default() -> Self {
        PhotonAccumulator {
            sum: Default::default(),
            weight_sum: 0.0
        }
    }
}
