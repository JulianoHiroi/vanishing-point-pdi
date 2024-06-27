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

def filter_lines(lines, intensity = 10):
    # para linhas com rhos e thetas muito próximos, mantenha apenas 10 linhas para cada grupo
    # e remova as linhas restantes
    #print(lines)
    lines = sorted(lines, key=lambda x: x[0][0])
    #print(lines)
    i = 0
    while i < len(lines) - 1:
        if np.abs(lines[i][0][0] - lines[i + 1][0][0]) < intensity and np.abs(lines[i][0][1] - lines[i + 1][0][1]) < intensity/100:
            del lines[i + 1]
        else:
            i += 1

    return lines


def verificate_direction(perspective_image, sumrow, sumcol, col, row, vanishrow, vanishcol):

     change = 2

     col2 = col+sumcol
     row2 = row+sumrow

     if col+sumcol not in range(0, perspective_image.shape[1]):
         changecol = False
         col2 = col
         change = change - 1
     else:
         changecol = True

     if row+sumrow not in range(0, perspective_image.shape[0]):
         changerow = False
         row2 = row
         change = change - 1
     else:
         changerow = True
     
     # anda na direção onde pixel estiver branco
     if change != 0:
         if perspective_image[row][col2] == 255 and vanishcol != col and changecol == True:
             col = col + sumcol
         elif perspective_image[row2][col] == 255 and vanishrow != row and changerow == True:
             row = row + sumrow
         elif perspective_image[row2][col2] == 255 and vanishrow != row and vanishcol != col and changecol == True and changerow == True:
             row = row + sumrow
             col = col + sumcol

     return row, col, change


def calculate_direction(row, col, vanishrow, vanishcol):
    if vanishrow - row > 0:
        sumrow = 1
    elif vanishrow - row < 0:
        sumrow = -1
    else:
        sumrow = 0
    if vanishcol - col > 0:
        sumcol = 1
    elif vanishcol - col < 0:
        sumcol = -1
    else:
        sumcol = 0

    return sumrow, sumcol
    

def detect_lines(edges, min_points = 100, filterLines = True):


    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_points)

    if filterLines == True:
      while len(lines) < 50: 
         min_points = min_points - 10
         lines = cv2.HoughLines(edges, 1, np.pi / 180, min_points)
      if(len(lines) > 200):
         lines = filter_lines(lines, 20)
      else:
         lines = filter_lines(lines)

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

def draw_lines(lines, filePath, image, color = True):
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
        if color == True:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            cv2.line(image, (x1, y1), (x2, y2), (255), 1)
    cv2.imwrite('resultados/image_lines_'+ filePath , image)


# gera imagem com linhas passando pelo ponto de fuga
def generate_perspective_image(vanishing_point, imageBorda, filePath):
   # gera imagem preta
    perspective_image = np.zeros((imageBorda.shape[0], imageBorda.shape[1]), np.uint8)

    # coloca um pixel branco em cada ponto de fuga
    for vanishing in vanishing_point:
      perspective_image[int(vanishing[1])][int(vanishing[0])] = 255

    lines = detect_lines(perspective_image, 0, False)
    draw_lines(lines, "perspective_" + filePath, imageBorda)

    return perspective_image, lines


#  calcula tamanho da borda
def calculate_border(vanishing_point, image_shape):
    
    # calcula tamanho das bordas de acordo com o número de pontos
    if len(vanishing_point) > 1:
        left = abs(int(vanishing_point[0][0]))
        right = int(vanishing_point[1][0] - image_shape[1]) + 1
        bottom = int(max(vanishing_point[0][1] - image_shape[0], vanishing_point[1][1] - image_shape[0], 0)+5)
        top = int(abs(min(vanishing_point[0][1], vanishing_point[1][1], 0))+5)
    else:
        left = 0
        right = 0
        bottom = int(max(vanishing_point[0][1] - image_shape[0], 0))+5
        top = int(abs(min(vanishing_point[0][1], 0)))+5

    # calcula posições do ponto de fuga na imagem com borda
    for vanishing in vanishing_point:
        if vanishing[0] < 0:
            vanishing[0] = 0
        else:
            vanishing[0] = vanishing[0] + left

        vanishing[1] = vanishing[1] + top

    return top, bottom, left, right, vanishing_point


def put_object(perspective_image, vanishing_point, lines, filePath, image_object):

    vanishcol = vanishing_point[0][0]
    vanishrow = vanishing_point[0][1]

    line_point = False
    while line_point == False: 
    
            # sorteia posição onde objeto vai começar (horizontal)
            startRow = random.sample(range(0, 100), 1)
            startRow = startRow[0]
        
            # sorteia duas linhas quaisquer
            line1, line2 = random.sample(list(lines), 2)
        
            draw_lines([line1, line2], filePath, perspective_image, False)
        
            for i in range(startRow, perspective_image.shape[0]):
                for j in range(0, perspective_image.shape[1]):
                    if perspective_image[i][j] == 255:
                        line_point = True
                        break
                if line_point == True:
                   break
 
    row = i
    col = j
    # primeiro ponto 
    first_point = [col, row]

    # calcula para qual direção está o ponto de fuga
    sumrow, sumcol = calculate_direction(row, col, vanishrow, vanishcol)
    perspective_image[row][col] = 50
    i = 0

    # sorteia o tamanho do objeto
    tam = random.sample(range(50, 200), 1)

    # enquanto objeto não estiver com o tamanho sorteado OU objeto não estiver encostando no ponto de fuga
    while i!=tam[0] and (vanishrow != row and vanishcol != col ):
        if col+sumcol in range(0, perspective_image.shape[1]) or row+sumrow in range(0, perspective_image.shape[0]):
            
            row, col, verificate = verificate_direction(perspective_image, sumrow, sumcol, col, row, vanishrow, vanishcol)
            # verifica se ouve deslocamento ou não
            if verificate == 0:
                break
        else:
            break
        
        perspective_image[row][col] = 50
        i = i+1
    perspective_image[row][col] = 50

    # segundo ponto
    second_point = [col, row]
    
    # calcula ponto do objeto vertical ao primeiro ponto encontrado
    row = 0
    col = first_point[0]
    while row != perspective_image.shape[0]:
        if perspective_image[row][col] == 255 and col not in range(first_point[1]-3, first_point[1]+3,):
            break
        row = row + 1
    if row == perspective_image.shape[0]:
        row = row - 1
    # terceiro ponto 
    third_point = [col, row]
    perspective_image[row][col] = 50

    # calcula para qual direção está o ponto de fuga
    sumrow, sumcol = calculate_direction(row, col, vanishrow, vanishcol)
    perspective_image[row][col] = 50
    i = 0

    # enquanto objeto não estiver na coluna do segundo ponto OU objeto não estiver encostando no ponto de fuga
    while col!=second_point[0] and (vanishrow != row and vanishcol != col ):
        
        if col+sumcol in range(0, perspective_image.shape[1]) or row+sumrow in range(0, perspective_image.shape[0]):
            row, col, verificate = verificate_direction(perspective_image, sumrow, sumcol, col, row, vanishrow, vanishcol)

            # verifica se ouve deslocamento ou não
            if verificate == 0:
                break
        else:
            break
        
        perspective_image[row][col] = 25
        i = i+1

        #  se deu ruim
        if i == 1000:
            break

    if row == vanishrow or col == vanishcol:
        row = row+1
        col = col+1
    fourth_point = [col, row]
    perspective_image[row][col] = 25
    cv2.circle(perspective_image, ((first_point[0]), first_point[1]), 5, (50), -1)
    cv2.circle(perspective_image, ((second_point[0]), second_point[1]), 5, (50), -1)
    cv2.circle(perspective_image, ((third_point[0]), third_point[1]), 5, (50), -1)
    cv2.circle(perspective_image, ((fourth_point[0]), fourth_point[1]), 5, (50), -1)
    cv2.imwrite('resultados/sort_line_points'+ filePath , perspective_image)
    points = np.array([first_point, second_point, fourth_point, third_point])
    cv2.fillPoly(image_object, [points], color=(255,0,0))
    cv2.imwrite('resultados/perspective_image_with_object'+ filePath , image_object) 


# Função para encontrar o ponto de fuga
# para isso ele irá identificar os pontos e categorizando em 3 lugares diferente (direita, esquerda e centro)
def ransac_vanishing_point(lines, num_iterations, threshold, width):
    best_intersection = np.zeros((3, 2))  # Inicializar um array 3x2 de zeros para as melhores interseções
    best_count = np.zeros(3, dtype=int)  # Inicializar um array de zeros para as melhores contagens

    for _ in range(num_iterations):
        line1, line2 = random.sample(list(lines), 2)
        intersection = compute_intersection(line1, line2)
        if intersection is None:
            continue

        # Converta a interseção para um formato 1D se estiver em formato de coluna
        intersection = intersection.flatten()

        count = 0
        for line in lines:
            if distance_point_to_line(intersection, line) < threshold:
                count += 1
        
        # Atualizar best_count e best_intersection de acordo com a posição da interseção
        if intersection[0] < 0 and count > best_count[0]:
            best_count[0] = count
            best_intersection[0] = intersection
        elif intersection[0] > width and count > best_count[2]:
            best_count[2] = count
            best_intersection[2] = intersection
        elif 0 <= intersection[0] <= width and count > best_count[1]:
            best_count[1] = count
            best_intersection[1] = intersection
    print("Melhor contagem: ", best_count)
    left, mid , right = best_count
    if(left*2 > mid or right*2 > mid):
        best_count[1] = 0
    else:
        best_count[0] = 0
        best_count[2] = 0

    
    
    best_count_not_zero = best_count != 0
    return best_intersection[best_count_not_zero]


def vanishing_point(filePath):
    
    # Carregar imagem
    imagelines = cv2.imread("imagens/" +  filePath)
    image = imagelines.copy()
    image_object = image.copy()
    
    # Detectar bordas
    edges = detect_edges(image)
    cv2.imwrite('resultados/image_edges_'+ filePath , edges)
    
    
    # Detectar linhas
    lines = detect_lines(edges)

    cv2.imwrite('resultados/image_edges_'+ filePath , edges)
    #Desenha as linhas na imagem
    draw_lines(lines, filePath ,  imagelines) 

    # Definir parâmetros do RANSAC
    num_iterations = 1000
    threshold = 5
    
    # Encontrar ponto de fuga
    vanishing_point = ransac_vanishing_point(lines, num_iterations, threshold, image.shape[1])

    vanishing_point = np.array(vanishing_point)

    original_vanishing_point = vanishing_point

    top, bottom, left, right, vanishing_point = calculate_border(vanishing_point, image.shape)

    imageBorda = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    for vanishing in vanishing_point:
        cv2.circle(imageBorda, (int(vanishing[0]), int(vanishing[1])) , 10, (0, 0, 255), -1)

    cv2.imwrite('resultados/ponto1' + filePath , imageBorda)

    perspective_image, lines = generate_perspective_image(vanishing_point, imageBorda, filePath)
    

    if len(vanishing_point) == 1:
        put_object(perspective_image, original_vanishing_point, lines, filePath, image_object)



def main ():


    
    try:
        # Remove the directory 'resultados' if it exists
        if os.path.exists('resultados'):
            shutil.rmtree('resultados')
        
        # Create the directory 'resultados'
        os.mkdir('resultados')

    except OSError as e:
        print(f"Error: {e}")

    images = [ "image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpg", "image6.jpg", "image7.jpg", "image8.jpg" , "image9.jpg", "image10.jpg"]
    #images = ["image3.jpeg"]

    # crie o diretório resultados
   
   
    for(image) in images:
        print("Vanishing Point da imagem " + image + " : ")
        vanishing_point(image)
    

if __name__ == '__main__':

    main()