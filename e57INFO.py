import os
import numpy as np
from PIL import Image
import pye57
import time
import laspy


def save_points_to_laz(points, output_path):
    """
    Сохранение точек в LAZ-файл (LAS 1.4) с нормалями, интенсивностью и цветом.
    """
    header = laspy.LasHeader(point_format=2, version="1.4")
    header.scales = np.array([0.000001, 0.000001, 0.000001])
    header.offsets = np.array([0.0, 0.0, 0.0])

    las = laspy.LasData(header)
    if len(points) == 0:
        print("Нет точек для сохранения")
        return

    las.X = points['X']
    las.Y = points['Y']
    las.Z = points['Z']
    las.intensity = points['intensity'].astype(np.uint16)
    las.red   = (points['normal_r'].astype(np.uint16) * 256)
    las.green = (points['normal_g'].astype(np.uint16) * 256)
    las.blue  = (points['normal_b'].astype(np.uint16) * 256)

    las.write(output_path)
    print(f"LAZ сохранён: {output_path}")


if __name__ == '__main__':
    e57_path = r"S:\Depo\testNorm\А.e57"
    laz_out = os.path.splitext(e57_path)[0] + "_with_normals.laz"

    e57 = pye57.E57(e57_path)
    header = e57.get_header(0)
    sensor_origin = np.array(header.translation, dtype=np.float32)
    print("Позиция скана (sensor origin):", sensor_origin)

    t0 = time.perf_counter()
    data = e57.read_scan(0, intensity=True, row_column=True, ignore_missing_fields=True)
    X = data['cartesianX']; Y = data['cartesianY']; Z = data['cartesianZ']
    intensity = data['intensity']
    col = data['columnIndex']; row = data['rowIndex']

    width = int(col.max()) + 1
    height = int(row.max()) + 1

    # Усреднение координат по ячейкам
    X_sum = np.zeros((height, width), dtype=np.float32)
    Y_sum = np.zeros((height, width), dtype=np.float32)
    Z_sum = np.zeros((height, width), dtype=np.float32)
    cnt   = np.zeros((height, width), dtype=np.int32)
    np.add.at(X_sum, (row, col), X)
    np.add.at(Y_sum, (row, col), Y)
    np.add.at(Z_sum, (row, col), Z)
    np.add.at(cnt,   (row, col), 1)

    mask = cnt > 0
    X_img = np.where(mask, X_sum / cnt, np.nan)
    Y_img = np.where(mask, Y_sum / cnt, np.nan)
    Z_img = np.where(mask, Z_sum / cnt, np.nan)

    # Векторное вычисление нормалей
    u_x = X_img[:, 2:] - X_img[:, :-2]
    u_y = Y_img[:, 2:] - Y_img[:, :-2]
    u_z = Z_img[:, 2:] - Z_img[:, :-2]
    v_x = X_img[2:, :] - X_img[:-2, :]
    v_y = Y_img[2:, :] - Y_img[:-2, :]
    v_z = Z_img[2:, :] - Z_img[:-2, :]

    # Кросс-произведение
    Nx = u_y[1:-1, :] * v_z[:, 1:-1] - u_z[1:-1, :] * v_y[:, 1:-1]
    Ny = u_z[1:-1, :] * v_x[:, 1:-1] - u_x[1:-1, :] * v_z[:, 1:-1]
    Nz = u_x[1:-1, :] * v_y[:, 1:-1] - u_y[1:-1, :] * v_x[:, 1:-1]

    # Нормализация
    norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    valid = norm > 0
    Nx[valid] /= norm[valid]
    Ny[valid] /= norm[valid]
    Nz[valid] /= norm[valid]

    # Подготовка полных массивов и маски внутренних точек
    full_Nx = np.zeros((height, width), dtype=np.float32)
    full_Ny = np.zeros((height, width), dtype=np.float32)
    full_Nz = np.zeros((height, width), dtype=np.float32)
    inner = (
        mask[1:-1, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:] &
        mask[:-2, 1:-1] & mask[2:, 1:-1]
    )
    full_Nx[1:-1, 1:-1][inner] = Nx[inner]
    full_Ny[1:-1, 1:-1][inner] = Ny[inner]
    full_Nz[1:-1, 1:-1][inner] = Nz[inner]

    # Ориентация к сенсору
    Xc = X_img - sensor_origin[0]
    Yc = Y_img - sensor_origin[1]
    Zc = Z_img - sensor_origin[2]
    dot = full_Nx * Xc + full_Ny * Yc + full_Nz * Zc
    flip = dot > 0
    full_Nx[flip] *= -1
    full_Ny[flip] *= -1
    full_Nz[flip] *= -1

    # Мэппинг нормалей в uint8
    normal_r = np.zeros_like(full_Nx, dtype=np.uint8)
    normal_g = np.zeros_like(full_Ny, dtype=np.uint8)
    normal_b = np.zeros_like(full_Nz, dtype=np.uint8)
    valid_full = (full_Nx != 0) | (full_Ny != 0) | (full_Nz != 0)
    normal_r[valid_full] = np.round((full_Nx[valid_full] * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_g[valid_full] = np.round((full_Ny[valid_full] * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_b[valid_full] = np.round((full_Nz[valid_full] * 0.5 + 0.5) * 255).astype(np.uint8)

    # Сохранение JPG нормалей
    normals_img = np.flipud(np.dstack((normal_r, normal_g, normal_b)))
    img_norm = os.path.splitext(e57_path)[0] + "_normals.jpg"
    Image.fromarray(normals_img).save(img_norm)
    print(f"JPG нормалей: {img_norm}")

    # Формирование точек для LAZ
    rows_idx, cols_idx = np.where(inner)
    rows_idx += 1; cols_idx += 1
    n_pts = len(rows_idx)
    pts = np.zeros(n_pts, dtype=[('X', 'i8'), ('Y', 'i8'), ('Z', 'i8'),
                                  ('intensity', 'u2'), ('normal_r', 'u1'),
                                  ('normal_g', 'u1'), ('normal_b', 'u1')])


    header.scales = np.array([0.000001, 0.000001, 0.000001])

    pts['X'] = np.round(X_img[rows_idx, cols_idx] / header.scales[0]).astype(np.int64)
    pts['Y'] = np.round(Y_img[rows_idx, cols_idx] / header.scales[1]).astype(np.int64)
    pts['Z'] = np.round(Z_img[rows_idx, cols_idx] / header.scales[2]).astype(np.int64)
    intensity_img = intensity.reshape((height, width))
    pts['intensity'] = intensity_img[rows_idx, cols_idx].astype(np.uint16)
    pts['normal_r'] = normal_r[rows_idx, cols_idx]
    pts['normal_g'] = normal_g[rows_idx, cols_idx]
    pts['normal_b'] = normal_b[rows_idx, cols_idx]

    save_points_to_laz(pts, laz_out)
    print(f"Общее время: {time.perf_counter() - t0:.3f} с")
