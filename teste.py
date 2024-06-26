import cv2
import numpy as np
import random
import os
import shutil

def detect_edges(image):
    area = image.shape[0]*image.shape[1]

    threshold = 40
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, threshold, threshold*3, apertureSize=3)
    i = 0
    for i in range(10):
        i = i + 1
       
        edges = cv2.Canny(blur, threshold, threshold*3, apertureSize=3)
        soma = np.sum(edges)/255
        if soma > area*0.04:
            
            threshold = threshold + threshold//(2.3**i) + 1
        elif soma < area*0.03:
            
            threshold = threshold - threshold//(2**i) + 1
        else:
            break
    #print("A porcentagem de borda da imagem é: ", soma/area*100, "%")
    return edges

def filter_lines(lines):
    # para linhas com rhos e thetas muito próximos, mantenha apenas 10 linhas para cada grupo
    # e remova as linhas restantes
    #print(lines)
    lines = sorted(lines, key=lambda x: x[0][0])
    #print(lines)
    i = 0
    while i < len(lines) - 1:
        if np.abs(lines[i][0][0] - lines[i + 1][0][0]) < 20 and np.abs(lines[i][0][1] - lines[i + 1][0][1]) < 0.2:
            del lines[i + 1]
        else:
            i += 1

    return lines


def detect_lines(edges, min_points = 100):


    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_points)
    print("Número de linhas detectadas: ", len(lines))
    lines = filter_lines(lines)
    print("Número de linhas detectadas: ", len(lines))

    # faça um assert para garantir que tenha mais de 2 linhas e manda msg se não tiver
    assert lines is not None, "Não foi possível detectar linhas na imagem"
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

def distance_point_to_line(point, equation_line):
    rho, theta = equation_line[0]
    x0, y0 = point

    a = np.cos(theta)
    b = np.sin(theta)

    return np.abs(a * x0 + b * y0 - rho) / np.sqrt(a**2 + b**2)

def draw_lines(lines, filePath, image):
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
    cv2.imwrite('resultados2/image_lines_'+ filePath , image)

def ransac_vanishing_point(lines, num_iterations, threshold , width):

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


    assert best_intersection is not None, "Não foi possível encontrar o ponto de fuga"
    return best_intersection 

#restrição para a direita
def restrictRight(intersection, width):
    if intersection[0] > width//2 :
        return True
    return False

# restringe para a esquerda
def restrictLeft(intersection, width):
    if intersection[0] < width//2:
        return True
    return False

# restringe para cima
def restrictTop(intersection, height):
    if intersection[1] < height//2:
        return True
    return False
#restringe para baixo
def restrictBottom(intersection, height):
    if intersection[1] > height//2:
        return True
    return False


def vanishing_point(filePath):
    # Carregar imagem
    imagelines = cv2.imread("imagens/" +  filePath)
    image = imagelines.copy()
    
    # Detectar bordas
    edges = detect_edges(image)
    cv2.imwrite('resultados2/image_edges_'+ filePath , edges)
    
    
    # Detectar linhas
    lines = detect_lines(edges)
    cv2.imwrite('resultados2/image_edges_'+ filePath , edges)
    #Desenha as linhas na imagem
    draw_lines(lines, filePath ,  imagelines) 
    # Definir parâmetros do RANSAC
    num_iterations = 1000
    threshold = 5
    
    # Encontrar ponto de fuga
    print("Vanishing Point da imagem " + filePath + " : ")
    vanishing_point = ransac_vanishing_point(lines, num_iterations, threshold , image.shape[0] )
    
    cv2.circle(image, (int(vanishing_point[0][0]), int(vanishing_point[1][0])), 10, (0, 0, 255), -1)
    cv2.imwrite('resultados2/ponto1' + filePath , image)
    '''
    # Faz borda brancas
    imageBorda = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.circle(imageBorda, (int(vanishing_point[0][0] + 200), int(vanishing_point[1][0] + 200)), 10, (0, 0, 255), -1)
    cv2.circle(imageBorda, (int(vanishing_point2[0][0] + 200), int(vanishing_point2[1][0] + 200)), 10, (0, 0, 255), -1)
    cv2.imwrite('resultados2/image_borda_pontos' + filePath , imageBorda)'''
    



def main ():


    
    try:
        # Remove the directory 'resultados2' if it exists
        if os.path.exists('resultados2'):
            shutil.rmtree('resultados2')
        
        # Create the directory 'resultados2'
        os.mkdir('resultados2')

    except OSError as e:
        print(f"Error: {e}")

    images = ["01.jpg", "02.jpg", "image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg", "image8.jpg"]
    #images = ["image5.jpg"]

    # crie o diretório resultados2
   
   
    for(image) in images:
        vanishing_point(image)
    

if __name__ == '__main__':

    main()