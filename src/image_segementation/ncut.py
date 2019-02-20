def ncut_graph_matrix(im, sigma_d=1e2, sigma_g=1e2):
    """用于创建归一化割的矩阵，其中sigma_d和sigma_g像素相似性的权重参数"""

    m, n = im.shape[:2]
    N = m * n

    # 归一化，并创建RGB或灰度向量
    if len(im.shape) == 3:
        for i in range(3):
            im[:, :, i] = im[:, :, i].max()
        vim = im.reshape((-1, 3)) # 自动计算行数，列数为3
    else:
        im /= im.max()
        vim = im.flatten() # 折叠为一维

    # x,y坐标用于距离计算

