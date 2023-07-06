use std::collections::HashMap;

use crate::triangle::Triangle;
use nalgebra::{Matrix4, Vector3, Vector4};

#[allow(dead_code)]
pub enum Buffer {
    Color,
    Depth,
    Both,
}

#[allow(dead_code)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Default, Clone)]
pub struct Rasterizer {
    model: Matrix4<f64>,
    view: Matrix4<f64>,
    projection: Matrix4<f64>,
    pos_buf: HashMap<usize, Vec<Vector3<f64>>>,
    ind_buf: HashMap<usize, Vec<Vector3<usize>>>,
    col_buf: HashMap<usize, Vec<Vector3<f64>>>,

    frame_buf: Vec<Vector3<f64>>,
    depth_buf: Vec<f64>,
    //MSAA
    frame_sample: Vec<Vector3<f64>>,
    depth_sample: Vec<f64>,
    width: u64,
    height: u64,
    next_id: usize,
}

#[derive(Clone, Copy)]
pub struct PosBufId(usize);

#[derive(Clone, Copy)]
pub struct IndBufId(usize);

#[derive(Clone, Copy)]
pub struct ColBufId(usize);

impl Rasterizer {
    pub fn new(w: u64, h: u64) -> Self {
        let mut r = Rasterizer::default();
        r.width = w;
        r.height = h;
        r.frame_buf.resize((w * h) as usize, Vector3::zeros());
        r.depth_buf.resize((w * h) as usize, 0.0);
        r.frame_sample
            .resize((w * h * 4) as usize, Vector3::zeros());
        r.depth_sample.resize((w * h * 4) as usize, 0.0);
        r
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        ((self.height - 1 - y as u64) * self.width + x as u64) as usize
    }

    fn get_super_index(&self, x: usize, y: usize, x_offset: usize, y_offset: usize) -> usize {
        ((self.height * 2 - 1 - y as u64 * 2 - y_offset as u64) * self.width * 2
            + x as u64 * 2
            + x_offset as u64) as usize
    }

    fn set_pixel(&mut self, point: &Vector3<f64>, color: &Vector3<f64>) {
        let ind = (self.height as f64 - 1.0 - point.y) * self.width as f64 + point.x;
        self.frame_buf[ind as usize] = *color;
    }

    fn set_super_pixel(
        &mut self,
        x: usize,
        y: usize,
        x_offset: usize,
        y_offset: usize,
        color: &Vector3<f64>,
    ) {
        let idx = self.get_super_index(x, y, x_offset, y_offset);
        self.frame_sample[idx] = *color;
    }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
            }
            Buffer::Depth => {
                self.depth_buf.fill(f64::MAX);
                self.depth_sample.fill(f64::MAX);
            }
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MAX);
                self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_sample.fill(f64::MAX);
            }
        }
    }

    pub fn set_model(&mut self, model: Matrix4<f64>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f64>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f64>) {
        self.projection = projection;
    }

    fn get_next_id(&mut self) -> usize {
        let res = self.next_id;
        self.next_id += 1;
        res
    }

    pub fn load_position(&mut self, positions: &Vec<Vector3<f64>>) -> PosBufId {
        let id = self.get_next_id();
        self.pos_buf.insert(id, positions.clone());
        PosBufId(id)
    }

    pub fn load_indices(&mut self, indices: &Vec<Vector3<usize>>) -> IndBufId {
        let id = self.get_next_id();
        self.ind_buf.insert(id, indices.clone());
        IndBufId(id)
    }

    pub fn load_colors(&mut self, colors: &Vec<Vector3<f64>>) -> ColBufId {
        let id = self.get_next_id();
        self.col_buf.insert(id, colors.clone());
        ColBufId(id)
    }

    pub fn draw(
        &mut self,
        pos_buffer: PosBufId,
        ind_buffer: IndBufId,
        col_buffer: ColBufId,
        _typ: Primitive,
    ) {
        let buf = &self.clone().pos_buf[&pos_buffer.0];
        let ind: &Vec<Vector3<usize>> = &self.clone().ind_buf[&ind_buffer.0];
        let col = &self.clone().col_buf[&col_buffer.0];

        let f1 = (50.0 - 0.1) / 2.0;
        let f2 = (50.0 + 0.1) / 2.0;

        let mvp = self.projection * self.view * self.model;

        for i in ind {
            let mut t = Triangle::new();
            let mut v = vec![
                mvp * to_vec4(buf[i[0]], Some(1.0)), // homogeneous coordinates
                mvp * to_vec4(buf[i[1]], Some(1.0)),
                mvp * to_vec4(buf[i[2]], Some(1.0)),
            ];

            for vec in v.iter_mut() {
                *vec = *vec / vec.w;
            }
            for vert in v.iter_mut() {
                vert.x = 0.5 * self.width as f64 * (vert.x + 1.0);
                vert.y = 0.5 * self.height as f64 * (vert.y + 1.0);
                vert.z = vert.z * f1 + f2;
            }
            for j in 0..3 {
                // t.set_vertex(j, Vector3::new(v[j].x, v[j].y, v[j].z));
                t.set_vertex(j, v[j].xyz());
            }
            let col_x = col[i[0]];
            let col_y = col[i[1]];
            let col_z = col[i[2]];
            t.set_color(0, col_x[0], col_x[1], col_x[2]);
            t.set_color(1, col_y[0], col_y[1], col_y[2]);
            t.set_color(2, col_z[0], col_z[1], col_z[2]);

            self.rasterize_triangle(&t);
        }
    }

    pub fn rasterize_triangle(&mut self, t: &Triangle) {
        let v = &t.to_vector4();
        let x_min = v[0].x.min(v[1].x).min(v[2].x) as usize;
        let x_max = v[0].x.max(v[1].x).max(v[2].x) as usize;
        let y_min = v[0].y.min(v[1].y).min(v[2].y) as usize;
        let y_max = v[0].y.max(v[1].y).max(v[2].y) as usize;
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                let mut flag = false;
                for x_offset in 0..2 {
                    for y_offset in 0..2 {
                        let x_sample = x as f64 + 0.25 + x_offset as f64 * 0.5;
                        let y_sample = y as f64 + 0.25 + x_offset as f64 * 0.5;
                        if inside_triangle(x_sample, y_sample, &t.v) {
                            let (c1, c2, c3) = compute_barycentric2d(x_sample, y_sample, &t.v);
                            let z_interpolated = (c1 * v[0].z / v[0].w
                                + c2 * v[1].z / v[1].w
                                + c3 * v[2].z / v[2].w)
                                / (c1 / v[0].w + c2 / v[1].w + c3 / v[2].w);
                            let idx = self.get_super_index(x, y, x_offset, y_offset);
                            if z_interpolated < self.depth_sample[idx] {
                                flag = true;
                                self.depth_sample[idx] = z_interpolated;
                                self.set_super_pixel(x, y, x_offset, y_offset, &t.get_color());
                            }
                        }
                    }
                }
                if flag {
                    let mut color = Vector3::zeros();
                    for x_offset in 0..2 {
                        for y_offset in 0..2 {
                            color +=
                                self.frame_sample[self.get_super_index(x, y, x_offset, y_offset)];
                        }
                    }
                    color[0] /= 4.0;
                    color[1] /= 4.0;
                    color[2] /= 4.0;
                    self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &color)
                }
            }
        }
    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }
}

fn to_vec4(v3: Vector3<f64>, w: Option<f64>) -> Vector4<f64> {
    Vector4::new(v3.x, v3.y, v3.z, w.unwrap_or(1.0))
}

fn inside_triangle(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> bool {
    let p = Vector3::new(x, y, 0.0);
    let ap = p - v[0];
    let bp = p - v[1];
    let cp = p - v[2];
    let ab = v[1] - v[0];
    let bc = v[2] - v[1];
    let ca = v[0] - v[2];
    let res1 = ab.cross(&ap);
    let res2 = bc.cross(&bp);
    let res3 = ca.cross(&cp);
    (res1.z < 0.0 && res2.z < 0.0 && res3.z < 0.0) || (res1.z > 0.0 && res2.z > 0.0 && res3.z > 0.0)
}

fn compute_barycentric2d(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> (f64, f64, f64) {
    let c1 = (x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * y + v[1].x * v[2].y - v[2].x * v[1].y)
        / (v[0].x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * v[0].y + v[1].x * v[2].y
            - v[2].x * v[1].y);
    let c2 = (x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * y + v[2].x * v[0].y - v[0].x * v[2].y)
        / (v[1].x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * v[1].y + v[2].x * v[0].y
            - v[0].x * v[2].y);
    let c3 = (x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * y + v[0].x * v[1].y - v[1].x * v[0].y)
        / (v[2].x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * v[2].y + v[0].x * v[1].y
            - v[1].x * v[0].y);
    (c1, c2, c3)
}
