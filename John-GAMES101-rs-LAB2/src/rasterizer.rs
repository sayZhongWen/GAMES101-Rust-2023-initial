use std::collections::HashMap;

use crate::triangle::Triangle;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

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
    // frame_sample: Vec<Vector3<f64>>,
    // depth_sample: Vec<f64>,
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
        // r.frame_sample
        //     .resize((w * h * 4) as usize, Vector3::zeros());
        // r.depth_sample.resize((w * h * 4) as usize, 0.0);
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

    // fn set_super_pixel(
    //     &mut self,
    //     x: usize,
    //     y: usize,
    //     x_offset: usize,
    //     y_offset: usize,
    //     color: &Vector3<f64>,
    // ) {
    //     let idx = self.get_super_index(x, y, x_offset, y_offset);
    //     self.frame_sample[idx] = *color;
    // }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                // self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
            }
            Buffer::Depth => {
                self.depth_buf.fill(f64::MAX);
                // self.depth_sample.fill(f64::MAX);
            }
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MAX);
                // self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
                // self.depth_sample.fill(f64::MAX);
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
                if inside_triangle(x as f64 + 0.5, y as f64 + 0.5, &t.v) {
                    let (c1, c2, c3) = compute_barycentric2d(x as f64 + 0.5, y as f64 + 0.5, &t.v);
                    let z_interpolated =
                        (c1 * v[0].z / v[0].w + c2 * v[1].z / v[1].w + c3 * v[2].z / v[2].w)
                            / (c1 / v[0].w + c2 / v[1].w + c3 / v[2].w);
                    let idx = self.get_index(x, y);
                    if z_interpolated < self.depth_buf[idx] {
                        self.depth_buf[idx] = z_interpolated;
                        self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &t.get_color());
                    }
                }
            }
        }
        self.fxaa();
    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }

    pub fn get_luminance(&self) -> Vec<f64> {
        let mut res = Vec::with_capacity((self.width * self.height) as usize);
        for x in 0..self.width {
            for y in 0..self.height {
                res.push(0.0);
            }
        }
        for x in 0..self.width {
            for y in 0..self.height {
                res.push(0.0);
                let idx = self.get_index(x as usize, y as usize);
                res[idx] = luminance(&self.frame_buf[idx]);
            }
        }
        res
    }

    pub fn fxaa(&mut self) {
        const CONTRAST_THRESHOLD: f64 = 0.0833;
        const RELATIVE_THRESHOLD: f64 = 0.125;
        let luma: Vec<f64> = self.get_luminance();
        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                let n = luma[self.get_index(x as usize, y as usize + 1)];
                let w = luma[self.get_index(x as usize - 1, y as usize)];
                let e = luma[self.get_index(x as usize + 1, y as usize)];
                let s = luma[self.get_index(x as usize, y as usize - 1)];
                let m = luma[self.get_index(x as usize, y as usize)];
                let nw = luma[self.get_index(x as usize - 1, y as usize + 1)];
                let ne = luma[self.get_index(x as usize + 1, y as usize + 1)];
                let sw = luma[self.get_index(x as usize - 1, y as usize - 1)];
                let se = luma[self.get_index(x as usize + 1, y as usize - 1)];
                let max_luma = n.max(e).max(w).max(s).max(m);
                let min_luma = n.min(e).min(w).min(s).min(m);
                let contrast = max_luma - min_luma; //根据色彩的亮度对比度判断是否要模糊边界处理
                if contrast >= CONTRAST_THRESHOLD.max(RELATIVE_THRESHOLD * max_luma) {
                    let mut filter = (2.0 * (n + e + s + w) + ne + nw + se + sw) / 12.0;
                    filter = (filter - m).abs();
                    filter = saturate(filter / contrast);
                    let mut pixel_blend = smoothstep(0.0, 1.0, filter);
                    pixel_blend = pixel_blend * pixel_blend;
                    let vertical = (n + s - 2.0 * m).abs() * 2.0
                        + (ne + se - 2.0 * e).abs()
                        + (nw + sw - 2.0 * w).abs();
                    let horizontal = (e + w - 2.0 * m).abs() * 2.0
                        + (ne + nw - 2.0 * n).abs()
                        + (se + sw - 2.0 * s).abs();
                    let is_horizontal = vertical > horizontal;
                    let mut pixel_step = if is_horizontal {
                        Vector2::new(0.0, 1.0)
                    } else {
                        Vector2::new(1.0, 0.0)
                    };
                    let positive = (if is_horizontal { n } else { e }).abs();
                    let negative = (if is_horizontal { s } else { w }).abs();
                    if positive < negative {
                        pixel_step = -pixel_step;
                    }
                    let direction = Vector2::new(x as f64, y as f64) + pixel_step; //色彩混合方向
                    let final_color = (self.frame_buf[self.get_index(x as usize, y as usize)]
                        + self.frame_buf
                            [self.get_index(direction.x as usize, direction.y as usize)])
                        / 2.0; //为了让效果更明显一点（并且这张图并不复杂）我没有选择加权
                    self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &final_color);
                }
            }
        }
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
fn luminance(color: &Vector3<f64>) -> f64 {
    0.213 * color[0] + 0.715 * color[1] + 0.072 * color[2]
}
pub fn saturate(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x > 1.0 {
        return 1.0;
    }
    x
}
pub fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = saturate((x - a) / (b - a));
    t * t * (3.0 - (2.0 * t))
}
