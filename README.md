## Detector de Ponto de Fuga: Uma Abordagem Baseada em Canny, Hough e RANSAC

Este projeto implementa um detector de ponto de fuga, utilizando técnicas de processamento de imagem para identificar o ponto de fuga em imagens. O ponto de fuga é um ponto imaginário no espaço onde as linhas paralelas de um objeto 3D parecem convergir quando projetadas em uma imagem 2D. É um conceito fundamental em perspectiva, e a detecção precisa desse ponto é crucial em diversas aplicações, como:

- **Realidade aumentada:** Para sobrepor objetos 3D em imagens de forma realista, é necessário conhecer a perspectiva da imagem.
- **Reconstrução 3D:** O ponto de fuga é usado para reconstruir modelos 3D a partir de imagens 2D.
- **Calibração de câmera:** O ponto de fuga pode ser usado para calcular a posição da câmera em relação à cena.

O projeto é estruturado em três etapas principais:

**1. Detecção de Bordas com o Algoritmo de Canny:**

O primeiro passo é detectar as bordas dos objetos na imagem. Para isso, o algoritmo de Canny é aplicado. O algoritmo de Canny é um detector de bordas bem estabelecido, que consiste em três etapas:

- **Suavização:** A imagem é suavizada por um filtro gaussiano para reduzir o ruído e o efeito de detalhes irrelevantes.
- **Detecção de Gradiente:** A magnitude e a direção do gradiente da imagem são calculadas.
- **Supressão de Não-Máximos:** Os pontos de borda falsos são eliminados por meio da supressão de não-máximos, que identifica os pontos de maior gradiente ao longo da direção do gradiente.

**2. Extração de Linhas com a Transformada de Hough:**

Após a detecção de bordas, as linhas que compõem as estruturas da imagem são extraídas com a transformada de Hough. A transformada de Hough é uma técnica matemática que transforma a representação da imagem no espaço de parâmetros, onde as linhas são representadas por suas equações.

A transformada de Hough para linhas é definida pela equação:

```
ρ = x cos θ + y sin θ
```

Onde:

- **ρ (rho):** é a distância da origem à linha.
- **θ (theta):** é o ângulo da linha em relação ao eixo horizontal.
- **x, y:** são as coordenadas de um ponto na linha.

A transformada de Hough cria um espaço de parâmetros (ρ, θ), onde cada ponto representa uma linha possível na imagem. As linhas são detectadas pela identificação de picos no espaço de parâmetros, que correspondem a um grande número de pontos que se ajustam à mesma linha.

**3. Detecção do Ponto de Fuga com RANSAC:**

O ponto de fuga é o ponto onde as linhas paralelas da imagem convergem. Para determinar o ponto de fuga, o método RANSAC (Random Sample Consensus) é usado. O RANSAC é um algoritmo de robustês que identifica o melhor modelo a partir de dados corrompidos por valores discrepantes (outliers).

No contexto do projeto, o RANSAC escolhe aleatoriamente duas linhas e calcula seu ponto de intersecção. Em seguida, o algoritmo avalia a distância das demais linhas em relação a esse ponto. O ponto de intersecção que minimiza a distância média das linhas é considerado o ponto de fuga.

**Geração de Objetos Perspectivos:**

O projeto inclui um algoritmo adicional para gerar objetos 2D com formas retangulares que respeitam a perspectiva da imagem. Esses objetos são gerados utilizando o ponto de fuga detectado como referência. O algoritmo garante que as linhas do objeto convergem para o ponto de fuga, criando uma aparência realista de perspectiva.
