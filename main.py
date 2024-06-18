import cv2
import numpy as np
import random

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    return lines

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]

def distance_point_to_line(point, line):
    px, py = point
    x1, y1, x2, y2 = line[0]

    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / denom

def ransac_vanishing_point(lines, num_iterations, threshold):
    best_vanishing_point = None
    best_inliers = 0

    for _ in range(num_iterations):
        line1, line2 = random.sample(list(lines), 2)
        intersection = compute_intersection(line1, line2)
        if intersection is None:
            continue

        inliers = 0
        for line in lines:
            dist = distance_point_to_line(intersection, line)
            if dist < threshold:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_vanishing_point = intersection

    return best_vanishing_point

# Carregar imagem
image = cv2.imread('imagens/02.jpg')

# Detectar bordas
edges = detect_edges(image)

# Detectar linhas
lines = detect_lines(edges)

# Definir parâmetros do RANSAC
num_iterations = 1000
threshold = 5

# Encontrar ponto de fuga
vanishing_point = ransac_vanishing_point(lines, num_iterations, threshold)

if vanishing_point:
    print(f"Ponto de fuga: {vanishing_point}")
    cv2.circle(image, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (0, 255, 0), -1)

# Mostrar imagem com o ponto de fuga
cv2.imshow('Ponto de Fuga', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
