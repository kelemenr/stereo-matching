def write_to_point_cloud(width, height, image, focal_length, baseline, dmin,
                         scale, file_name, path):
    with open(path + file_name + ".xyz", "w+") as xyz_file:
        for i in range(height):
            for j in range(width):
                z = (baseline * focal_length) / (
                    int(image[i, j]) + dmin)
                x = (i * z) / focal_length
                y = (j * z) / focal_length
                xyz_file.write("{} {} {}\n".format(x, y, z / scale))
