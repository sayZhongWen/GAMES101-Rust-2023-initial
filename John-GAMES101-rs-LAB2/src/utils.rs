use nalgebra::{Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{cvt_color, COLOR_RGB2BGR};
use std::os::raw::c_void;

pub fn degrees_to_radians(a: f64) -> f64 {
    a * std::f64::consts::PI / 180.0
}

pub(crate) fn get_view_matrix(eye_pos: Vector3<f64>) -> Matrix4<f64> {
    let view = Matrix4::new(
        1.0, 0.0, 0.0, -eye_pos.x, 0.0, 1.0, 0.0, -eye_pos.y, 0.0, 0.0, 1.0, -eye_pos.z, 0.0, 0.0,
        0.0, 1.0,
    );
    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    let a = degrees_to_radians(rotation_angle);
    let model: Matrix4<f64> = Matrix4::new(
        a.cos(),
        -a.sin(),
        0.0,
        0.0,
        a.sin(),
        a.cos(),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    model
}

pub(crate) fn get_projection_matrix(
    eye_fov: f64,
    aspect_ratio: f64,
    z_near: f64,
    z_far: f64,
) -> Matrix4<f64> {
    let b = z_near.abs() * (eye_fov / 2.0).tan();
    let t = -b;
    let r = t * aspect_ratio;
    let l = -r;
    let projection = Matrix4::new(
        2.0 / (r - l),
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / (t - b),
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / (z_near - z_far),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ) * Matrix4::new(
        1.0,
        0.0,
        0.0,
        -(r + l) / 2.0,
        0.0,
        1.0,
        0.0,
        -(t + b) / 2.0,
        0.0,
        0.0,
        1.0,
        -(z_far + z_near) / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ) * Matrix4::new(
        z_near,
        0.0,
        0.0,
        0.0,
        0.0,
        z_near,
        0.0,
        0.0,
        0.0,
        0.0,
        z_near + z_far,
        -z_near * z_far,
        0.0,
        0.0,
        1.0,
        0.0,
    );
    projection
}

pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<Vector3<f64>>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700,
            700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        )
        .unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image
        .convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0)
        .expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}
