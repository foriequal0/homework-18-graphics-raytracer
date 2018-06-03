#[allow(dead_code)]
pub mod linsrgb {
    use palette::{LinSrgb, Srgb};

    pub fn black() -> LinSrgb { LinSrgb::new(0.0, 0.0, 0.0) }

    pub fn grey() -> LinSrgb { Srgb::new(0.5, 0.5, 0.5).into_linear() }

    pub fn white() -> LinSrgb { LinSrgb::new(1.0, 1.0, 1.0) }

    pub fn red() -> LinSrgb { LinSrgb::new(1.0, 0.0, 0.0) }

    pub fn green() -> LinSrgb { LinSrgb::new(0.0, 1.0, 0.0) }

    pub fn blue() -> LinSrgb { LinSrgb::new(0.0, 0.0, 1.0) }

    pub fn yellow() -> LinSrgb { LinSrgb::new(1.0, 1.0, 0.0) }

    pub fn cyan() -> LinSrgb { LinSrgb::new(0.0, 1.0, 1.0) }

    pub fn magenta() -> LinSrgb { LinSrgb::new(1.0, 0.0, 1.0) }
}