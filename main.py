import cv2
import numpy as np
import random

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    return edges

def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    return lines

def compute_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])

    det_A = np.linalg.det(A)
    if det_A != 0:
        intersection = np.linalg.solve(A, b)
        return intersection
    else:
        return None

    return intersection

def distance_point_to_line(point, equation_line):
    rho, theta = equation_line[0]
    x0, y0 = point

    a = np.cos(theta)
    b = np.sin(theta)

    return np.abs(a * x0 + b * y0 - rho) / np.sqrt(a**2 + b**2)



def ransac_vanishing_point(lines, num_iterations, threshold):

    best_intersection = None
    best_count = 0

    for _ in range(num_iterations):
        line1, line2 = random.sample(list(lines), 2)
        intersection = compute_intersection(line1, line2)
        if intersection is None:
            continue

        count = 0
        for line in lines:
            if distance_point_to_line(intersection, line) < threshold:
                count += 1

        if count > best_count:
            best_count = count
            best_intersection = intersection

    return best_intersection

def main ():
    # Carregar imagem
    image = cv2.imread('imagens/02.jpg')
    image2 = cv2.imread('imagens/02.jpg')
    # Detectar bordas
    edges = detect_edges(image)

    # Detectar linhas
    lines = detect_lines(edges)
    cv2.imshow('Bordas', edges)

    #Desenha as linhas na imagem
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow('Linhas', image) 
    
    # Definir parâmetros do RANSAC
    num_iterations = 1000
    threshold = 5

    # Encontrar ponto de fuga
    vanishing_point = ransac_vanishing_point(lines, num_iterations, threshold)
    
    # Desenhar ponto de fuga
    cv2.circle(image2, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (0, 0, 255), -1)

    cv2.imshow('Ponto de Fuga', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()