use std::os::raw::c_void;
use nalgebra::{Matrix3, Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub type V3d = Vector3<f64>;
pub fn degrees_to_radians(a:f64)->f64{
    a*std::f64::consts::PI/180.0
}
pub(crate) fn get_view_matrix(eye_pos: V3d) -> Matrix4<f64> {
    let mut view: Matrix4<f64> = Matrix4::identity();
    view=view*Matrix4::new(1.0,0.0,0.0,-eye_pos.x,
    0.0,1.0,0.0,-eye_pos.y,
    0.0,0.0,1.0,-eye_pos.z,
    0.0,0.0,0.0,1.0);
    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    let a=degrees_to_radians(rotation_angle);
    let model: Matrix4<f64> = Matrix4::new(a.cos(),-a.sin(),0.0,0.0,
                                               a.sin(),a.cos(),0.0,0.0,
    0.0,0.0,1.0,0.0,
    0.0,0.0,0.0,1.0);
    model
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Matrix4<f64> {
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
    ) *Matrix4::new(
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

pub(crate) fn get_arbitrary_rotation(axis:Vector3<f64>,angle:f64)->Matrix4<f64>{
    let radian=degrees_to_radians(angle);
    let x=axis.x;
    let y=axis.y;
    let z=axis.z;
    let rotation:Matrix3<f64>=radian.cos()*Matrix3::identity()+(1.0-radian.cos())*Matrix3::new(x*x,x*y,x*z,
    x*y,y*y,y*z,
    x*z,y*z,z*z)+radian.sin()*Matrix3::new(0.0,-z,y,
    z,0.0,-x,
    -y,x,0.0);
    Matrix4::new(rotation.m11,rotation.m12,rotation.m13,0.0,
    rotation.m21,rotation.m22,rotation.m23,0.0,
    rotation.m31,rotation.m32,rotation.m33,0.0,
    0.0,0.0,0.0,1.0)
}

pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<V3d>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700, 700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        ).unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image.convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0).expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}