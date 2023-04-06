use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use ndarray::{s, Array, Axis, Dim};
use ndarray_linalg::Eig;

fn main() {
    let file = "ColorHistogram.asc";
    let mut data = read_data(file);
    // remove the index column
    data.remove_index(Axis(1), 0);

    let data_var = data.var_axis(Axis(0), 1.);
    println!("Data variance: {:.4}", data_var.sum());

    let projected_data = pca(&data, 5);
    println!("Projected data: {:?}", projected_data.slice(s![..10, ..]));

    let pca_var = projected_data.var_axis(Axis(0), 1.);
    println!("PCA data variance: {:.4}", pca_var.sum());
}

fn pca(data: &Array<f64, Dim<[usize; 2]>>, num_components: usize) -> Array<f64, Dim<[usize; 2]>> {
    // 计算数据的均值
    let mean = data.mean_axis(Axis(0)).unwrap();

    // 将数据矩阵按列进行零均值化
    let centered = data - mean;

    // 计算数据的协方差矩阵
    let cov_matrix = centered.t().dot(&centered) / ((data.shape()[0] - 1) as f64);

    // 对协方差矩阵进行特征值分解
    let (_eigs, vecs) = cov_matrix.eig().unwrap();

    // 选取前 num_components 个主成分
    let principal_components = vecs.mapv(|x| x.re).slice_move(s![.., ..num_components]);

    // 将数据投影到选取的主成分上
    let projected_data = centered.dot(&principal_components);

    return projected_data;
}

fn read_data(filename: &str) -> Array<f64, Dim<[usize; 2]>> {
    // 打开文件并读取数据
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        data.push(row);
    }
    return Array::from_shape_vec((data.len(), data[0].len()), data.concat()).unwrap();
}
