#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    ssaa_factor = 1  # fator de supersampling atual (1 = sem SSAA)
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    transform_stack = []  # pilha para transforms aninhados
    is_animation = False  # flag para detectar se estamos em uma animação
    
    # Lighting system variables
    lights = []  # Lista de luzes direcionais
    headlight_enabled = True  # NavigationInfo headlight
    ambient_light = [0.2, 0.2, 0.2]  # Luz ambiente global
    _time_sensor_states = {}  # estados de TimeSensors (id->start_time)

    @staticmethod
    def calculate_lighting(world_pos, normal, material):
        """Calcula a iluminação para um ponto usando o modelo de iluminação Phong."""
        if not normal or len(normal) != 3:
            return material.get("diffuseColor", [0.8, 0.8, 0.8])
        
        # Normaliza a normal (que pessimo isso)
        norm_length = math.sqrt(sum(n*n for n in normal)) or 1.0
        normal = [n / norm_length for n in normal]
        
        # Extrai propriedades do material
        diffuse = material.get("diffuseColor", [0.8, 0.8, 0.8])
        specular = material.get("specularColor", [0.0, 0.0, 0.0])
        emissive = material.get("emissiveColor", [0.0, 0.0, 0.0])
        shininess = material.get("shininess", 0.2)
        ambient_intensity = material.get("ambientIntensity", 0.2)
        
        # Se não há luzes, usa apenas ambient + emissive
        if not GL.lights:
            ambient = [GL.ambient_light[i] * diffuse[i] * ambient_intensity for i in range(3)]
            final_color = [emissive[i] + ambient[i] for i in range(3)]
            final_color = [max(0.0, min(1.0, c)) for c in final_color]
            return final_color
        
        # Cor ambiente (global + material)
        ambient = [GL.ambient_light[i] * diffuse[i] * ambient_intensity for i in range(3)]
        # Cor final acumulada inicia com emissive + ambient
        final_color = [emissive[i] + ambient[i] for i in range(3)]
        
        # Vetor do observador para specular (do ponto até câmera)
        # O vetor observador precisa apontar do ponto na superfície para a câmera (observador) 
        # para que o produto escalar com o vetor refletido represente o cosseno do ângulo entre os dois.
        if hasattr(GL, 'camera_position'):
            view_dir = [GL.camera_position[i] - world_pos[i] for i in range(3)]
            vlen = math.sqrt(sum(v*v for v in view_dir)) or 1.0
            view_dir = [v / vlen for v in view_dir]
        else:
            view_dir = [0.0, 0.0, 1.0]

        # Calcular contribuição de cada luz
        for light in GL.lights:
            if light['type'] in ['directional', 'headlight']:
                # Direção da luz (para DirectionalLight é fixa, para headlight é a direção da câmera)
                light_dir = light['direction']
                
                # Componente ambiente da luz
                la = light.get('ambientIntensity', 0.0)
                for i in range(3):
                    final_color[i] += la * light['color'][i] * diffuse[i]
                
                # Produto escalar: normal . luz (máximo 0)
                dot_nl = max(0.0, sum(normal[i] * (-light_dir[i]) for i in range(3)))
                
                if dot_nl > 0:
                    # Componente difusa (cor principal do material)
                    for i in range(3):
                        final_color[i] += light['intensity'] * light['color'][i] * diffuse[i] * dot_nl
                    
                    # Componente especular (reflexos brilhantes)
                    if any(s > 0 for s in specular) and shininess > 0:
                        # Calcula direção de reflexão perfeita
                        reflect_dir = [2 * dot_nl * normal[i] - (-light_dir[i]) for i in range(3)]
                        rl = math.sqrt(sum(r*r for r in reflect_dir)) or 1.0
                        reflect_dir = [r/rl for r in reflect_dir]
                        
                        # Produto escalar entre reflexão e direção da câmera
                        dot_rv = max(0.0, sum(reflect_dir[i] * view_dir[i] for i in range(3)))
                        
                        # Fator especular (shininess controla a nitidez do brilho)
                        spec_exponent = max(1.0, shininess * 128.0)
                        spec_factor = pow(dot_rv, spec_exponent)
                        
                        # Adiciona reflexo especular
                        for i in range(3):
                            final_color[i] += light['intensity'] * light['color'][i] * specular[i] * spec_factor
        
        # Limita valores entre 0 e 1
        final_color = [max(0.0, min(1.0, c)) for c in final_color]
        return final_color

    @staticmethod
    def clear_lights():
        """Limpa todas as luzes da cena."""
        GL.lights = []

    @staticmethod
    def calculate_triangle_normal(v0, v1, v2):
        """Calcula a normal de um triângulo usando produto vetorial."""
        # Vetores das arestas
        edge1 = [v1[i] - v0[i] for i in range(3)]
        edge2 = [v2[i] - v0[i] for i in range(3)]
        
        # Produto vetorial edge1 x edge2
        normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ]
        
        # Normaliza
        length = math.sqrt(sum(n*n for n in normal)) or 1.0
        return [n / length for n in normal]

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        # Reset lighting system for each scene
        # GL.lights = []  # COMENTADO: não limpar luzes - elas são configuradas pelo X3D
        GL.headlight_enabled = True
        
        # Melhora qualidade para animações aplicando supersampling leve
        # Aumenta resolução interna em 50% para reduzir pixelização
        if hasattr(GL, '_is_animation') and GL._is_animation:
            GL.ssaa_factor = 1.5
            GL.width = int(width * GL.ssaa_factor)
            GL.height = int(height * GL.ssaa_factor)
        else:
            GL.ssaa_factor = 1

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Conversão de cor emissiva (0..1) para (0..255)
        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        r, g, b = [max(0, min(255, int(c * 255))) for c in emissive]
        f = getattr(GL, 'ssaa_factor', 1)
        # Desenha cada ponto escalando pelo fator de SSAA para utilizar o framebuffer ampliado
        for i in range(0, len(point), 2):
            x = int(round(point[i] * f))
            y = int(round(point[i + 1] * f))
            if 0 <= x < GL.width and 0 <= y < GL.height:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        if len(lineSegments) < 4:
            return

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        col = [max(0, min(255, int(c * 255))) for c in emissive]

        f = getattr(GL, 'ssaa_factor', 1)
        # Converte lista em lista de pontos inteiros escalados
        pts = []
        for i in range(0, len(lineSegments), 2):
            x = int(round(lineSegments[i] * f))
            y = int(round(lineSegments[i + 1] * f))
            pts.append((x, y))

        # com algoritmo de Bresenham pra corrigir o erro acumulado
        def draw_line(p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while True:
                if 0 <= x0 < GL.width and 0 <= y0 < GL.height:
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, col)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

        for i in range(len(pts) - 1):
            draw_line(pts[i], pts[i + 1])

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        if radius is None or radius <= 0:
            return
        f = getattr(GL, 'ssaa_factor', 1)
        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        col = [max(0, min(255, int(c * 255))) for c in emissive]
        # Centro na tela (usando resolução ampliada já presente em GL.width/height)
        cx = GL.width // 2
        cy = GL.height // 2
        r = int(round(radius * f))
        # Algoritmo do círculo de ponto médio
        x = r
        y = 0
        d = 1 - r
        def put(px, py):
            if 0 <= px < GL.width and 0 <= py < GL.height:
                gpu.GPU.draw_pixel([px, py], gpu.GPU.RGB8, col)
        while x >= y:
            put(cx + x, cy + y); put(cx + y, cy + x)
            put(cx - y, cy + x); put(cx - x, cy + y)
            put(cx - x, cy - y); put(cx - y, cy - x)
            put(cx + y, cy - x); put(cx + x, cy - y)
            y += 1
            if d < 0:
                d += 2 * y + 1
            else:
                x -= 1
                d += 2 * (y - x) + 1


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        if len(vertices) < 6:
            return
        f = getattr(GL, 'ssaa_factor', 1)
        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        col = [max(0, min(255, int(c * 255))) for c in emissive]

        def draw_filled_triangle(p0, p1, p2):
            (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
            # reduzindo iterações
            min_x = max(0, min(x0, x1, x2))
            max_x = min(GL.width - 1, max(x0, x1, x2))
            min_y = max(0, min(y0, y1, y2))
            max_y = min(GL.height - 1, max(y0, y1, y2))

            # Área dupla do triângulo
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if area == 0:
                return  # Dava errado

            # Pré-cálculo para incremento baricentrico
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    w0 = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
                    w1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                    w2 = (x0 - x2) * (y - y2) - (y0 - y2) * (x - x2)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, col)

        # Percorre cada triângulo (3 vértices -> 6 valores)
        for i in range(0, len(vertices), 6):
            if i + 5 >= len(vertices):
                break
            p0 = (int(round(vertices[i] * f)),     int(round(vertices[i + 1] * f)))
            p1 = (int(round(vertices[i + 2] * f)), int(round(vertices[i + 3] * f)))
            p2 = (int(round(vertices[i + 4] * f)), int(round(vertices[i + 5] * f)))
            draw_filled_triangle(p0, p1, p2)
            

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        if not point or len(point) < 9:
            return

        transp = colors.get("transparency", 0.0) if colors else 0.0

        def transform_vertex(v):
            # transforma um vértice 3D do espaço do objeto até a tela (pixel).
            # Retorna (x_screen, y_screen, z_ndc, inv_w, world_pos)
            vec = np.array([v[0], v[1], v[2], 1.0])
            world_pos = v[:]  # Posição original no espaço do objeto
            
            if hasattr(GL, 'model_matrix'):
                vec = GL.model_matrix @ vec
                world_pos = (GL.model_matrix @ np.array([v[0], v[1], v[2], 1.0]))[:3].tolist()
            if hasattr(GL, 'view_matrix'):
                vec = GL.view_matrix @ vec
            if hasattr(GL, 'projection_matrix'):
                vec = GL.projection_matrix @ vec
            w = vec[3] if vec[3] != 0 else 1.0
            inv_w = 1.0 / w
            # NDC
            x_ndc = vec[0] * inv_w
            y_ndc = vec[1] * inv_w
            z_ndc = vec[2] * inv_w  # assumindo z em [-1,1]
            x = int(round((1.0 - (x_ndc * 0.5 + 0.5)) * (GL.width - 1)))
            y = int(round((1.0 - (y_ndc * 0.5 + 0.5)) * (GL.height - 1)))
            return (x, y, z_ndc, inv_w, world_pos)

        def depth_test_and_write(x, y, z):
            # Otimização: pula depth test para objetos sem materiais especiais
            if not colors or (not colors.get("specularColor", [0,0,0]) and not colors.get("transparency", 0)):
                return True  # Aceita pixel sem teste para performance
            try:
                current = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
            except Exception:
                return True
            depth = (z + 1.0) * 0.5
            if depth < current:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [depth])
                return True
            return False

        def blend(dst, src, alpha):
            return [int(src[i] * (1-alpha) + dst[i] * alpha) for i in range(3)]

        for i in range(0, len(point), 9):
            if i + 8 >= len(point):
                break
            v0 = [point[i], point[i+1], point[i+2]]
            v1 = [point[i+3], point[i+4], point[i+5]]
            v2 = [point[i+6], point[i+7], point[i+8]]
            
            # Calcula a normal do triângulo no espaço mundial
            triangle_normal = GL.calculate_triangle_normal(v0, v1, v2)
            
            # Transforma normal para o espaço mundial
            world_normal = triangle_normal
            if hasattr(GL, 'model_matrix'):
                # Para transformar normais, usamos a transposta da inversa da matriz de modelo
                # Simplificação: assumimos que a matriz não tem escala não-uniforme
                try:
                    # Matriz 3x3 da parte superior esquerda da model_matrix
                    upper_3x3 = GL.model_matrix[:3, :3]
                    normal_transform = np.linalg.inv(upper_3x3).T
                    normal_vec = np.array(triangle_normal)
                    world_normal = (normal_transform @ normal_vec)
                    norm_length = np.linalg.norm(world_normal) or 1.0
                    world_normal = (world_normal / norm_length).tolist()
                except np.linalg.LinAlgError:
                    # Se a matriz não é inversível, use transformação simples
                    normal_vec = np.array([triangle_normal[0], triangle_normal[1], triangle_normal[2], 0.0])
                    world_normal = (GL.model_matrix @ normal_vec)[:3]
                    norm_length = np.linalg.norm(world_normal) or 1.0
                    world_normal = (world_normal / norm_length).tolist()
            
            p0 = transform_vertex(v0)
            p1 = transform_vertex(v1)
            p2 = transform_vertex(v2)
            
            # Preenche o triângulo usando a regra Top-Left para evitar falhas nas bordas
            def draw_filled_triangle(p0, p1, p2):
                (x0, y0, z0, w0, world0), (x1, y1, z1, w1, world1), (x2, y2, z2, w2, world2) = p0, p1, p2
                # Bounding box half-open [min, max)
                min_x = max(0, int(math.floor(min(x0, x1, x2))))
                max_x = min(GL.width, int(math.ceil(max(x0, x1, x2))) )
                min_y = max(0, int(math.floor(min(y0, y1, y2))))
                max_y = min(GL.height, int(math.ceil(max(y0, y1, y2))) )

                def edge(ax, ay, bx, by, px, py):
                    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

                def is_top_left(ax, ay, bx, by):
                    dx = bx - ax
                    dy = by - ay
                    return (dy > 0) or (dy == 0 and dx < 0)

                area = edge(x0, y0, x1, y1, x2, y2)
                if area == 0:
                    return
                # Orientação consistente (CCW)
                if area < 0:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                    z1, z2 = z2, z1
                    w1, w2 = w2, w1
                    world1, world2 = world2, world1
                    area = -area

                topLeft0 = is_top_left(x1, y1, x2, y2)
                topLeft1 = is_top_left(x2, y2, x0, y0)
                topLeft2 = is_top_left(x0, y0, x1, y1)

                eps = 0.0  # com amostragem no centro, não precisamos de epsilon
                
                # Otimização inteligente com prioridade para qualidade em animações:
                # - Animações SEMPRE em qualidade máxima (step=1) para evitar pixelização
                # - Objetos transparentes e especulares também em qualidade máxima
                # - Apenas objetos estáticos muito simples usam step=2
                use_quality_mode = (
                    GL.is_animation or  # SEMPRE qualidade máxima para animações
                    (colors and colors.get("transparency", 0) > 0) or  # Objetos transparentes
                    (colors and colors.get("specularColor", [0,0,0]) != [0,0,0])  # Materiais especulares
                )
                
                # Para animações, SEMPRE usar step=1 (qualidade máxima)
                if GL.is_animation:
                    step = 1  # Força qualidade máxima para animações
                else:
                    step = 1 if use_quality_mode else 2  # Outros objetos podem usar step=2
                
                for y in range(min_y, max_y, step):
                    for x in range(min_x, max_x, step):
                        px = x + 0.5
                        py = y + 0.5
                        w0 = edge(x1, y1, x2, y2, px, py)
                        w1 = edge(x2, y2, x0, y0, px, py)
                        w2 = edge(x0, y0, x1, y1, px, py)
                        if (w0 > eps or (w0 >= -eps and topLeft0)) and \
                           (w1 > eps or (w1 >= -eps and topLeft1)) and \
                           (w2 > eps or (w2 >= -eps and topLeft2)):
                            # baricêntricas não normalizadas -> normalizar
                            wA = w0 / area
                            wB = w1 / area
                            wC = w2 / area
                            
                            z_interp = wA * z0 + wB * z1 + wC * z2
                            
                            # Otimização de depth test: pula para objetos simples não-animados
                            skip_depth_test = (
                                not use_quality_mode and  # Só pula se não estiver em modo qualidade
                                not colors or (
                                    not colors.get("transparency", 0) and 
                                    colors.get("specularColor", [0,0,0]) == [0,0,0]
                                )
                            )
                            
                            if skip_depth_test or depth_test_and_write(x, y, z_interp):
                                # Interpola posição mundial para iluminação por pixel
                                world_pos = [
                                    world0[i]*wA + world1[i]*wB + world2[i]*wC
                                    for i in range(3)
                                ]
                                
                                # Calcula iluminação correta
                                if GL.lights and len(GL.lights) > 0:
                                    # Usa sistema de iluminação Phong completo
                                    lit_color = GL.calculate_lighting(world_pos, world_normal, colors or {})
                                    final_col = [max(0, min(255, int(c * 255))) for c in lit_color]
                                else:
                                    # Sem luzes - usa cor emissiva ou difusa com iluminação ambiente mínima
                                    if colors and "emissiveColor" in colors:
                                        base_color = colors["emissiveColor"]
                                    elif colors and "diffuseColor" in colors:
                                        base_color = colors["diffuseColor"]
                                    else:
                                        base_color = [0.8, 0.8, 0.8]  # cinza padrão
                                    
                                    # Adiciona luz ambiente mínima
                                    ambient_factor = 0.3
                                    final_color = [base_color[i] + GL.ambient_light[i] * ambient_factor for i in range(3)]
                                    final_col = [max(0, min(255, int(c * 255))) for c in final_color]
                                
                                if transp > 0.0:
                                    dst = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                    final_col = blend(dst, final_col, transp)
                                
                                # Desenha com step para otimização (apenas para objetos simples)
                                if step > 1:
                                    for dy in range(step):
                                        for dx in range(step):
                                            px_fill = x + dx
                                            py_fill = y + dy
                                            if px_fill < GL.width and py_fill < GL.height:
                                                gpu.GPU.draw_pixel([px_fill, py_fill], gpu.GPU.RGB8, final_col)
                                else:
                                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, final_col)
            draw_filled_triangle(p0, p1, p2)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Coleta parâmetros do Viewpoint e monta as matrizes de câmera e projeção."""

        # guarda
        GL.camera_position = position
        GL.camera_orientation = orientation
        GL.camera_fov = fieldOfView

        # ----- axis-angle -> rotation matrix (reuses logic from transform_in) -----
        def axis_angle_to_matrix(ax):
            if not ax or len(ax) != 4:
                return np.identity(4)
            x, y, z, theta = ax
            c = math.cos(theta)
            s = math.sin(theta)
            n = math.sqrt(x*x + y*y + z*z) or 1.0
            x, y, z = x/n, y/n, z/n
            return np.array([
                [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s, 0],
                [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s, 0],
                [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c),   0],
                [0,                 0,               0,               1]
            ])

        # rotate default forward (-Z) and up (+Y)
        R = axis_angle_to_matrix(orientation)
        fwd = (R @ np.array([0, 0, -1, 0]))[:3]
        up  = (R @ np.array([0, 1,  0, 0]))[:3]
        eye = np.array(position)
        center = (eye + fwd).tolist()
        up = up.tolist()

        def look_at(eye, center, up):
            f = np.array(center) - np.array(eye)
            f = f / (np.linalg.norm(f) or 1.0)
            s = np.cross(up, f)
            s = s / (np.linalg.norm(s) or 1.0)
            u = np.cross(f, s) # na base ortonormal
            m = np.identity(4)
            # monta a matriz que leva o ponto eye até a origem
            m[0, :3] = s
            m[1, :3] = u
            m[2, :3] = -f
            m[:3, 3] = -np.dot(m[:3, :3], np.array(eye))
            return m

        GL.view_matrix = look_at(eye, center, up)
        # Armazena direção forward da câmera para uso pelo headlight
        GL.camera_forward = (fwd / (np.linalg.norm(fwd) or 1.0)).tolist()

        # projeção
        aspect = GL.width / GL.height
        near, far = GL.near, GL.far
        fovy = fieldOfView if fieldOfView and fieldOfView <= math.pi else math.radians(fieldOfView or 60)
        f = 1.0 / math.tan(fovy / 2)
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        GL.projection_matrix = proj

        # Atualiza (ou cria) direção do headlight para seguir a câmera
        if GL.headlight_enabled:
            headlight = None
            for l in GL.lights:
                if l.get('type') == 'headlight':
                    headlight = l
                    break
            if headlight is None:
                headlight = {
                    'type': 'headlight',
                    'ambientIntensity': 0.0,
                    'color': [1.0, 1.0, 1.0],
                    'intensity': 1.0,
                    'direction': [0.0, 0.0, -1.0]
                }
                GL.lights.append(headlight)
            # atualiza direção
            headlight['direction'] = GL.camera_forward
        
    @staticmethod
    def transform_in(translation, scale, rotation):
        """Empilha transform atual e aplica nova (suporte a transforms aninhados)."""
        # Monta T, R, S
        t = np.identity(4)
        if translation:
            t[:3, 3] = translation[:3]
        s_mat = np.identity(4)
        if scale:
            s_mat[0, 0] = scale[0]
            s_mat[1, 1] = scale[1]
            s_mat[2, 2] = scale[2]
        r_mat = np.identity(4)
        if rotation and len(rotation) == 4:
            x, y, z, theta = rotation
            c = math.cos(theta)
            s_ = math.sin(theta)
            n = math.sqrt(x*x + y*y + z*z) or 1.0
            x, y, z = x/n, y/n, z/n
            r_mat = np.array([
                [c + x*x*(1-c),     x*y*(1-c) - z*s_, x*z*(1-c) + y*s_, 0],
                [y*x*(1-c) + z*s_,  c + y*y*(1-c),    y*z*(1-c) - x*s_, 0],
                [z*x*(1-c) - y*s_,  z*y*(1-c) + x*s_, c + z*z*(1-c),    0],
                [0,                 0,               0,                1]
            ])
        local = t @ r_mat @ s_mat  # T * R * S
        prev = getattr(GL, 'model_matrix', np.identity(4))
        GL.transform_stack.append(prev)
        GL.model_matrix = prev @ local

    @staticmethod
    def transform_out():
        """Sai de um nó Transform: desempilha a matriz anterior."""
        if GL.transform_stack:
            GL.model_matrix = GL.transform_stack.pop()
        else:
            if hasattr(GL, 'model_matrix'):
                delattr(GL, 'model_matrix')

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Renderiza TriangleStripSet (tiras sequenciais de triângulos)."""
        if not point or not stripCount:
            return

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        # Material básico
        material = colors if colors else {}
        diffuse = material.get('diffuseColor', emissive)
        col = [max(0, min(255, int(c * 255))) for c in diffuse]

        # lista de vértices 3D
        verts = []
        for i in range(0, len(point), 3):
            verts.append([point[i], point[i+1], point[i+2]])

        # Pré-calcula normais por vértice (acumuladas) para todas as tiras
        vert_normals = [[0.0,0.0,0.0] for _ in verts]
        idx_offset = 0
        # Construir índices completos das tiras primeiro
        strip_vertices = []  # lista de listas com índices de cada tira
        cursor = 0
        for count in stripCount:
            indices = list(range(cursor, cursor+count))
            strip_vertices.append(indices)
            cursor += count
        # Para cada tira acumulamos as normais de cada triângulo
        for indices in strip_vertices:
            for i in range(len(indices)-2):
                i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
                if i % 2 == 1:  # alternância de orientação
                    i0, i1 = i1, i0
                v0 = verts[i0]; v1 = verts[i1]; v2 = verts[i2]
                e1 = [v1[j]-v0[j] for j in range(3)]
                e2 = [v2[j]-v0[j] for j in range(3)]
                n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]
                # acumula
                for vid in (i0,i1,i2):
                    vert_normals[vid][0] += n[0]
                    vert_normals[vid][1] += n[1]
                    vert_normals[vid][2] += n[2]
        # Normaliza
        for n in vert_normals:
            l = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) or 1.0
            n[0]/=l; n[1]/=l; n[2]/=l

        def transform_vertex(v):
            vec = np.array([v[0], v[1], v[2], 1.0])
            if hasattr(GL, 'model_matrix'):
                vec = GL.model_matrix @ vec
            if hasattr(GL, 'view_matrix'):
                vec = GL.view_matrix @ vec
            if hasattr(GL, 'projection_matrix'):
                vec = GL.projection_matrix @ vec
            if vec[3] != 0:
                vec = vec / vec[3]
            x = int(round((1.0 - (vec[0] * 0.5 + 0.5)) * (GL.width - 1)))
            y = int(round((1.0 - (vec[1] * 0.5 + 0.5)) * (GL.height - 1)))
            return (x, y, vec[2], v)

        def fill_triangle(p0, p1, p2, c0, c1, c2):
            (x0, y0, z0, v0), (x1, y1, z1, v1), (x2, y2, z2, v2) = p0, p1, p2
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(GL.width, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(GL.height, int(math.ceil(max(y0, y1, y2))))

            def edge(ax, ay, bx, by, px, py):
                return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

            def is_top_left(ax, ay, bx, by):
                dx = bx - ax
                dy = by - ay
                return (dy > 0) or (dy == 0 and dx < 0)

            area = edge(x0, y0, x1, y1, x2, y2)
            if area == 0:
                return
            if area < 0:
                x1, y1, x2, y2 = x2, y2, x1, y1
                area = -area

            topLeft0 = is_top_left(x1, y1, x2, y2)
            topLeft1 = is_top_left(x2, y2, x0, y0)
            topLeft2 = is_top_left(x0, y0, x1, y1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    px = x + 0.5
                    py = y + 0.5
                    w0 = edge(x1, y1, x2, y2, px, py)
                    w1 = edge(x2, y2, x0, y0, px, py)
                    w2 = edge(x0, y0, x1, y1, px, py)
                    if (w0 > 0 or (w0 == 0 and topLeft0)) and \
                       (w1 > 0 or (w1 == 0 and topLeft1)) and \
                       (w2 > 0 or (w2 == 0 and topLeft2)):
                        # Usa normal de face para lighting se houver luz
                        # Interpola cores (Gouraud)
                        # Usa áreas baricêntricas para pesos
                        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                        if area == 0:
                            continue
                        w0 = (x1 - x) * (y2 - y) - (y1 - y) * (x2 - x)
                        w1 = (x2 - x) * (y0 - y) - (y2 - y) * (x0 - x)
                        w2 = (x0 - x) * (y1 - y) - (y0 - y) * (x1 - x)
                        w0 /= area; w1 /= area; w2 /= area
                        R = c0[0]*w0 + c1[0]*w1 + c2[0]*w2
                        Gc = c0[1]*w0 + c1[1]*w1 + c2[1]*w2
                        Bc = c0[2]*w0 + c1[2]*w1 + c2[2]*w2
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [int(R), int(Gc), int(Bc)])

        base = 0
        for count in stripCount:
            if count < 3:
                base += count
                continue
            for i in range(count - 2):
                i0 = base + i
                i1 = base + i + 1
                i2 = base + i + 2
                if i % 2 == 0:
                    order = (i0, i1, i2)
                else:
                    order = (i1, i0, i2)
                p0 = transform_vertex(verts[order[0]])
                p1 = transform_vertex(verts[order[1]])
                p2 = transform_vertex(verts[order[2]])
                if GL.lights:
                    # Calcula cor iluminada por vértice
                    def lit_color(vid):
                        n = vert_normals[vid]
                        vpos = verts[vid]
                        lc = GL.calculate_lighting(vpos, n, material)
                        return [max(0, min(255, int(c*255))) for c in lc]
                    c0 = lit_color(order[0])
                    c1 = lit_color(order[1])
                    c2 = lit_color(order[2])
                else:
                    c0 = c1 = c2 = col
                fill_triangle(p0, p1, p2, c0, c1, c2)
            base += count

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Renderiza IndexedTriangleStripSet (-1 separa tiras)."""
        if not point or not index:
            return

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        material = colors if colors else {}
        diffuse = material.get('diffuseColor', emissive)
        col = [max(0, min(255, int(c * 255))) for c in diffuse]

        verts = []
        for i in range(0, len(point), 3):
            verts.append([point[i], point[i+1], point[i+2]])

        # Construir tiras (listas de índices) a partir de index separado por -1
        strips = []
        current = []
        for idx in index:
            if idx == -1:
                if len(current) >= 3:
                    strips.append(current)
                current = []
            else:
                if 0 <= idx < len(verts):
                    current.append(idx)
        if len(current) >= 3:
            strips.append(current)

        # Normais por vértice acumuladas
        vert_normals = [[0.0,0.0,0.0] for _ in verts]
        for strip in strips:
            for i in range(len(strip)-2):
                i0,i1,i2 = strip[i], strip[i+1], strip[i+2]
                if i % 2 == 1:
                    i0,i1 = i1,i0
                v0 = verts[i0]; v1 = verts[i1]; v2 = verts[i2]
                e1 = [v1[j]-v0[j] for j in range(3)]
                e2 = [v2[j]-v0[j] for j in range(3)]
                n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]
                for vid in (i0,i1,i2):
                    vert_normals[vid][0]+=n[0]
                    vert_normals[vid][1]+=n[1]
                    vert_normals[vid][2]+=n[2]
        for n in vert_normals:
            l = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) or 1.0
            n[0]/=l; n[1]/=l; n[2]/=l

        def transform_vertex(v):
            vec = np.array([v[0], v[1], v[2], 1.0])
            if hasattr(GL, 'model_matrix'):
                vec = GL.model_matrix @ vec
            if hasattr(GL, 'view_matrix'):
                vec = GL.view_matrix @ vec
            if hasattr(GL, 'projection_matrix'):
                vec = GL.projection_matrix @ vec
            if vec[3] != 0:
                vec = vec / vec[3]
            x = int(round((1.0 - (vec[0] * 0.5 + 0.5)) * (GL.width - 1)))
            y = int(round((1.0 - (vec[1] * 0.5 + 0.5)) * (GL.height - 1)))
            return (x, y, vec[2], v)

        def fill_triangle(p0, p1, p2, c0, c1, c2):
            (x0, y0, z0, v0), (x1, y1, z1, v1), (x2, y2, z2, v2) = p0, p1, p2
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(GL.width, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(GL.height, int(math.ceil(max(y0, y1, y2))))

            def edge(ax, ay, bx, by, px, py):
                return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

            def is_top_left(ax, ay, bx, by):
                dx = bx - ax
                dy = by - ay
                return (dy > 0) or (dy == 0 and dx < 0)

            area = edge(x0, y0, x1, y1, x2, y2)
            if area == 0:
                return
            if area < 0:
                x1, y1, x2, y2 = x2, y2, x1, y1

            topLeft0 = is_top_left(x1, y1, x2, y2)
            topLeft1 = is_top_left(x2, y2, x0, y0)
            topLeft2 = is_top_left(x0, y0, x1, y1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    px = x + 0.5
                    py = y + 0.5
                    w0 = edge(x1, y1, x2, y2, px, py)
                    w1 = edge(x2, y2, x0, y0, px, py)
                    w2 = edge(x0, y0, x1, y1, px, py)
                    if (w0 > 0 or (w0 == 0 and topLeft0)) and \
                       (w1 > 0 or (w1 == 0 and topLeft1)) and \
                       (w2 > 0 or (w2 == 0 and topLeft2)):
                        # Interpolação de cores Gouraud
                        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                        if area == 0:
                            continue
                        w0 = (x1 - x) * (y2 - y) - (y1 - y) * (x2 - x)
                        w1 = (x2 - x) * (y0 - y) - (y2 - y) * (x0 - x)
                        w2 = (x0 - x) * (y1 - y) - (y0 - y) * (x1 - x)
                        w0 /= area; w1 /= area; w2 /= area
                        R = c0[0]*w0 + c1[0]*w1 + c2[0]*w2
                        Gc = c0[1]*w0 + c1[1]*w1 + c2[1]*w2
                        Bc = c0[2]*w0 + c1[2]*w1 + c2[2]*w2
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [int(R), int(Gc), int(Bc)])

        for strip in strips:
            for i in range(len(strip)-2):
                i0,i1,i2 = strip[i], strip[i+1], strip[i+2]
                if i % 2 == 1:
                    i0,i1 = i1,i0
                order = (i0,i1,i2)
                p0 = transform_vertex(verts[order[0]])
                p1 = transform_vertex(verts[order[1]])
                p2 = transform_vertex(verts[order[2]])
                if GL.lights:
                    def lit_color(vid):
                        n = vert_normals[vid]
                        vpos = verts[vid]
                        lc = GL.calculate_lighting(vpos, n, material)
                        return [max(0, min(255, int(c*255))) for c in lc]
                    c0 = lit_color(order[0])
                    c1 = lit_color(order[1])
                    c2 = lit_color(order[2])
                else:
                    c0 = c1 = c2 = col
                fill_triangle(p0, p1, p2, c0, c1, c2)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Renderiza IndexedFaceSet triangulando em fan, com cores por vértice e textura simples."""
        if not coord or not coordIndex:
            return

        # vertices
        verts = []
        for i in range(0, len(coord), 3):
            verts.append([coord[i], coord[i+1], coord[i+2]])

        # faces como listas de índices, separadas por -1
        faces = []
        cur = []
        for idx in coordIndex:
            if idx == -1:
                if len(cur) >= 3:
                    faces.append(cur)
                cur = []
            else:
                if 0 <= idx < len(verts):
                    cur.append(idx)
        if len(cur) >= 3:
            faces.append(cur)

        # cores por vértice (opcional)
        face_colors = None
        if colorPerVertex and color and colorIndex:
            colors_list = []
            for i in range(0, len(color), 3):
                colors_list.append([color[i], color[i+1], color[i+2]])
            face_colors = []
            curc = []
            for ci in colorIndex:
                if ci == -1:
                    if len(curc) >= 3:
                        face_colors.append(curc)
                    curc = []
                else:
                    if 0 <= ci < len(colors_list):
                        curc.append(colors_list[ci])
            if len(curc) >= 3:
                face_colors.append(curc)
            if len(face_colors) != len(faces):
                face_colors = None

        # textura (opcional) com geração de mipmaps
        tex_faces = None
        mipmaps = None  # lista de níveis [level0, level1, ...]
        if current_texture and texCoord and texCoordIndex:
            try:
                base_img = gpu.GPU.load_texture(current_texture[0])
            except Exception:
                base_img = None
            if base_img is not None:
                # gera pirâmide até 1x1
                mipmaps = [base_img]
                lvl = 0
                current = base_img
                while current.shape[0] > 1 or current.shape[1] > 1:
                    h, w = current.shape[0], current.shape[1]
                    new_h = max(1, h // 2)
                    new_w = max(1, w // 2)
                    # média 2x2 simples (se ímpar, último pixel repete)
                    new_level = np.zeros((new_h, new_w, current.shape[2]), dtype=current.dtype)
                    for y in range(new_h):
                        for x in range(new_w):
                            block = current[y*2:min(h, y*2+2), x*2:min(w, x*2+2)]
                            new_level[y, x] = block.mean(axis=(0,1))
                    mipmaps.append(new_level)
                    current = new_level
                    lvl += 1
                # montar faces de UV
                uv_list = []
                for i in range(0, len(texCoord), 2):
                    uv_list.append([texCoord[i], texCoord[i+1]])
                tex_faces = []
                curu = []
                for ti in texCoordIndex:
                    if ti == -1:
                        if len(curu) >= 3:
                            tex_faces.append(curu)
                        curu = []
                    else:
                        if 0 <= ti < len(uv_list):
                            curu.append(uv_list[ti])
                if len(curu) >= 3:
                    tex_faces.append(curu)
                if len(tex_faces) != len(faces):
                    tex_faces = None

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0]) if colors else [1.0, 1.0, 1.0]
        emissive_col = [max(0, min(255, int(c * 255))) for c in emissive]
        transp = colors.get("transparency", 0.0) if colors else 0.0

        def transform_vertex(v):
            vec = np.array([v[0], v[1], v[2], 1.0])
            if hasattr(GL, 'model_matrix'):
                vec = GL.model_matrix @ vec
            if hasattr(GL, 'view_matrix'):
                vec = GL.view_matrix @ vec
            if hasattr(GL, 'projection_matrix'):
                vec = GL.projection_matrix @ vec
            w = vec[3] if vec[3] != 0 else 1.0
            inv_w = 1.0 / w
            x_ndc = vec[0] * inv_w
            y_ndc = vec[1] * inv_w
            z_ndc = vec[2] * inv_w
            x = int(round((1.0 - (x_ndc * 0.5 + 0.5)) * (GL.width - 1)))
            y = int(round((1.0 - (y_ndc * 0.5 + 0.5)) * (GL.height - 1)))
            return (x, y, z_ndc, inv_w)

        def choose_mip_level(uv0, uv1, uv2, p0, p1, p2):
            # Aproxima dU/dx, dV/dy via diferenças de tela -> heurística simples
            try:
                (x0, y0, *_), (x1, y1, *_), (x2, y2, *_) = p0, p1, p2
                du1 = uv1[0] - uv0[0]; dv1 = uv1[1] - uv0[1]
                du2 = uv2[0] - uv0[0]; dv2 = uv2[1] - uv0[1]
                dx1 = x1 - x0; dy1 = y1 - y0
                dx2 = x2 - x0; dy2 = y2 - y0
                # área em pixels
                area = abs(dx1*dy2 - dy1*dx2) + 1e-6
                # magnitude média de variação de UV
                du_avg = (abs(du1) + abs(du2)) * 0.5
                dv_avg = (abs(dv1) + abs(dv2)) * 0.5
                # estimativa粗: densidade de texels por pixel ~ (du_avg+dv_avg)/sqrt(area)
                density = (du_avg + dv_avg) / math.sqrt(area)
                level = max(0, math.log2(density * (mipmaps[0].shape[0] + mipmaps[0].shape[1]) * 0.25)) if density>0 else 0
                return int(min(level, len(mipmaps)-1))
            except Exception:
                return 0

        def sample_tex(u, v, level):
            # Retorna a cor do texel
            if mipmaps is None:
                return emissive_col
            level = int(max(0, min(level, len(mipmaps)-1)))
            tex = mipmaps[level]
            h, w = tex.shape[0], tex.shape[1]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            xi = int(u * (w - 1))
            yi = int((1 - v) * (h - 1))
            px = tex[yi, xi]
            if len(px) >= 3:
                return [int(px[0]), int(px[1]), int(px[2])]
            return emissive_col

        def depth_test_and_write(x, y, z_ndc):
            # Otimização: pula depth test para objetos simples
            if not colors or (not colors.get("specularColor", [0,0,0]) and not colors.get("transparency", 0)):
                return True  # Aceita pixel sem teste para performance
            try:
                current = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
            except Exception:
                return True
            depth = (z_ndc + 1.0) * 0.5
            if depth < current:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [depth])
                return True
            return False

        def blend(dst, src, alpha):
            return [int(src[i]*(1-alpha) + dst[i]*alpha) for i in range(3)]

        def draw_triangle(p0, p1, p2, c0, c1, c2, uv0, uv1, uv2):
            (x0, y0, z0, iw0), (x1, y1, z1, iw1), (x2, y2, z2, iw2) = p0, p1, p2
            min_x = max(0, int(math.floor(min(x0, x1, x2))))
            max_x = min(GL.width, int(math.ceil(max(x0, x1, x2))))
            min_y = max(0, int(math.floor(min(y0, y1, y2))))
            max_y = min(GL.height, int(math.ceil(max(y0, y1, y2))))

            def edge(ax, ay, bx, by, px, py):
                return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

            def is_top_left(ax, ay, bx, by):
                dx = bx - ax
                dy = by - ay
                return (dy > 0) or (dy == 0 and dx < 0)

            den = edge(x0, y0, x1, y1, x2, y2)
            if den == 0:
                return
            if den < 0:
                x1, y1, x2, y2 = x2, y2, x1, y1
                z1, z2 = z2, z1
                iw1, iw2 = iw2, iw1
                c1, c2 = c2, c1
                uv1, uv2 = uv2, uv1
                den = -den

            topLeft0 = is_top_left(x1, y1, x2, y2)
            topLeft1 = is_top_left(x2, y2, x0, y0)
            topLeft2 = is_top_left(x0, y0, x1, y1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    px = x + 0.5
                    py = y + 0.5
                    e0 = edge(x1, y1, x2, y2, px, py)
                    e1 = edge(x2, y2, x0, y0, px, py)
                    e2 = edge(x0, y0, x1, y1, px, py)
                    if (e0 > 0 or (e0 == 0 and topLeft0)) and \
                       (e1 > 0 or (e1 == 0 and topLeft1)) and \
                       (e2 > 0 or (e2 == 0 and topLeft2)):
                        wA = e0 / den
                        wB = e1 / den
                        wC = e2 / den
                        # perspective correct usando 1/w
                        inv_wA = wA * iw0
                        inv_wB = wB * iw1
                        inv_wC = wC * iw2
                        sum_inv = inv_wA + inv_wB + inv_wC
                        if sum_inv == 0:
                            continue
                        inv_norm = 1.0 / sum_inv
                        z_interp = (z0*inv_wA + z1*inv_wB + z2*inv_wC) * inv_norm
                        if not depth_test_and_write(x, y, z_interp):
                            continue
                        if uv0 is not None and uv1 is not None and uv2 is not None and mipmaps is not None:
                            u = (uv0[0]*inv_wA + uv1[0]*inv_wB + uv2[0]*inv_wC) * inv_norm
                            v = (uv0[1]*inv_wA + uv1[1]*inv_wB + uv2[1]*inv_wC) * inv_norm
                            lvl = choose_mip_level(uv0, uv1, uv2, p0, p1, p2)
                            col_px = sample_tex(u, v, lvl)
                        elif c0 is not None and c1 is not None and c2 is not None:
                            R = (c0[0]*inv_wA + c1[0]*inv_wB + c2[0]*inv_wC) * inv_norm
                            G = (c0[1]*inv_wA + c1[1]*inv_wB + c2[1]*inv_wC) * inv_norm
                            B = (c0[2]*inv_wA + c1[2]*inv_wB + c2[2]*inv_wC) * inv_norm
                            col_px = [int(max(0, min(255, R*255))), int(max(0, min(255, G*255))), int(max(0, min(255, B*255)))]
                        else:
                            # SHADING GARANTIDO - SEMPRE aplica shading independente de luzes
                            # Calcluando o shading da face mesmo que não tenha UVs
                            # Calcula orientação da face no espaço mundo
                            face_idx = [i0, i1, i2]
                            v0_world = verts[face_idx[0]]
                            v1_world = verts[face_idx[1]]
                            v2_world = verts[face_idx[2]]
                            
                            # Vetores das arestas no espaço mundo
                            edge1 = [v1_world[i] - v0_world[i] for i in range(3)]
                            edge2 = [v2_world[i] - v0_world[i] for i in range(3)]
                            
                            # Normal da face (produto vetorial)
                            normal = [
                                edge1[1]*edge2[2] - edge1[2]*edge2[1],
                                edge1[2]*edge2[0] - edge1[0]*edge2[2], 
                                edge1[0]*edge2[1] - edge1[1]*edge2[0]
                            ]
                            
                            # Normalizar
                            n_len = math.sqrt(sum(n*n for n in normal)) or 1.0
                            face_normal = [n/n_len for n in normal]
                            
                            # Sistema de iluminação Phong correto
                            if GL.lights and len(GL.lights) > 0:
                                # Para este pixel, estima posição no espaço mundial
                                # usando centro do triângulo como aproximação
                                center_world = [
                                    (v0_world[i] + v1_world[i] + v2_world[i]) / 3.0
                                    for i in range(3)
                                ]
                                
                                # Usa a normal da face para iluminação
                                lit_color = GL.calculate_lighting(center_world, face_normal, colors or {})
                                col_px = [max(0, min(255, int(c * 255))) for c in lit_color]
                            else:
                                # Sem luzes - usa cor do material com iluminação ambiente
                                if colors and "emissiveColor" in colors:
                                    base_color = colors["emissiveColor"]
                                elif colors and "diffuseColor" in colors:
                                    base_color = colors["diffuseColor"]
                                else:
                                    base_color = [0.8, 0.8, 0.8]
                                
                                # Adiciona luz ambiente mínima
                                ambient_factor = 0.3
                                final_color = [base_color[i] + GL.ambient_light[i] * ambient_factor for i in range(3)]
                                col_px = [max(0, min(255, int(c * 255))) for c in final_color]
                        if transp > 0.0:
                            dst = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                            col_px = blend(dst, col_px, transp)
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, col_px)

        for fi, face in enumerate(faces):
            base = face[0]
            for k in range(1, len(face) - 1):
                i0, i1, i2 = base, face[k], face[k+1]
                p0 = transform_vertex(verts[i0])
                p1 = transform_vertex(verts[i1])
                p2 = transform_vertex(verts[i2])
                c0 = c1 = c2 = None
                if face_colors and fi < len(face_colors):
                    fc = face_colors[fi]
                    try:
                        c0 = fc[ face.index(i0) ]
                        c1 = fc[ face.index(i1) ]
                        c2 = fc[ face.index(i2) ]
                    except Exception:
                        c0 = c1 = c2 = None
                uv0 = uv1 = uv2 = None
                if tex_faces and fi < len(tex_faces):
                    tf = tex_faces[fi]
                    try:
                        uv0 = tf[ face.index(i0) ]
                        uv1 = tf[ face.index(i1) ]
                        uv2 = tf[ face.index(i2) ]
                    except Exception:
                        uv0 = uv1 = uv2 = None
                draw_triangle(p0, p1, p2, c0, c1, c2, uv0, uv1, uv2)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        if not size or len(size) < 3:
            return
        
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        
        # Vértices do cubo (8 vértices)
        vertices = [
            [-sx, -sy, -sz],  # 0: Bottom-left-back
            [ sx, -sy, -sz],  # 1: Bottom-right-back
            [ sx,  sy, -sz],  # 2: Top-right-back
            [-sx,  sy, -sz],  # 3: Top-left-back
            [-sx, -sy,  sz],  # 4: Bottom-left-front
            [ sx, -sy,  sz],  # 5: Bottom-right-front
            [ sx,  sy,  sz],  # 6: Top-right-front
            [-sx,  sy,  sz],  # 7: Top-left-front
        ]
        
        # Faces do cubo (12 triângulos, 2 por face)
        faces = [
            # Back face (z = -sz)
            [0, 2, 1], [0, 3, 2],
            # Front face (z = sz)
            [4, 5, 6], [4, 6, 7],
            # Left face (x = -sx)
            [0, 4, 7], [0, 7, 3],
            # Right face (x = sx)
            [1, 6, 5], [1, 2, 6],
            # Bottom face (y = -sy)
            [0, 1, 5], [0, 5, 4],
            # Top face (y = sy)
            [3, 6, 2], [3, 7, 6],
        ]
        
        # Converte as faces em uma lista de pontos para o triangleSet
        points = []
        for face in faces:
            for vertex_idx in face:
                points.extend(vertices[vertex_idx])
        
        # Chama triangleSet para renderizar
        GL.triangleSet(points, colors)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        if not radius or radius <= 0:
            return
        
        # Parâmetros de tesselação
        latitudes = 16   # Divisões horizontais
        longitudes = 32  # Divisões verticais
        
        vertices = []
        
        # Gera vértices usando coordenadas esféricas
        for i in range(latitudes + 1):
            lat = math.pi * i / latitudes - math.pi / 2  # de -π/2 a π/2
            for j in range(longitudes):
                lng = 2 * math.pi * j / longitudes  # de 0 a 2π
                
                x = radius * math.cos(lat) * math.cos(lng)
                y = radius * math.sin(lat)
                z = radius * math.cos(lat) * math.sin(lng)
                vertices.append([x, y, z])
        
        # Gera triângulos
        points = []
        for i in range(latitudes):
            for j in range(longitudes):
                # Índices dos vértices para formar dois triângulos
                i0 = i * longitudes + j
                i1 = i * longitudes + (j + 1) % longitudes
                i2 = (i + 1) * longitudes + j
                i3 = (i + 1) * longitudes + (j + 1) % longitudes
                
                # Primeiro triângulo
                if i < latitudes:
                    points.extend(vertices[i0])
                    points.extend(vertices[i2])
                    points.extend(vertices[i1])
                
                # Segundo triângulo
                if i < latitudes:
                    points.extend(vertices[i1])
                    points.extend(vertices[i2])
                    points.extend(vertices[i3])
        
        # Chama triangleSet para renderizar
        GL.triangleSet(points, colors)

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        if not bottomRadius or not height or bottomRadius <= 0 or height <= 0:
            return
        
        # Parâmetros de tesselação
        segments = 32  # Divisões circulares
        
        vertices = []
        
        # Vértice do topo do cone
        top_vertex = [0, height / 2, 0]
        
        # Vértices da base
        base_vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = bottomRadius * math.cos(angle)
            z = bottomRadius * math.sin(angle)
            y = -height / 2
            base_vertices.append([x, y, z])
        
        # Centro da base
        base_center = [0, -height / 2, 0]
        
        points = []
        
        # Triângulos da lateral do cone
        for i in range(segments):
            i_next = (i + 1) % segments
            
            # Triângulo lateral
            points.extend(top_vertex)
            points.extend(base_vertices[i_next])
            points.extend(base_vertices[i])
        
        # Triângulos da base (fan triangulation)
        for i in range(segments):
            i_next = (i + 1) % segments
            
            # Triângulo da base
            points.extend(base_center)
            points.extend(base_vertices[i])
            points.extend(base_vertices[i_next])
        
        # Chama triangleSet para renderizar
        GL.triangleSet(points, colors)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        if not radius or not height or radius <= 0 or height <= 0:
            return
        
        # Parâmetros de tesselação
        segments = 32  # Divisões circulares
        
        vertices = []
        
        # Vértices do topo
        top_vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = height / 2
            top_vertices.append([x, y, z])
        
        # Vértices da base
        bottom_vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = -height / 2
            bottom_vertices.append([x, y, z])
        
        # Centros das faces
        top_center = [0, height / 2, 0]
        bottom_center = [0, -height / 2, 0]
        
        points = []
        
        # Triângulos da lateral do cilindro
        for i in range(segments):
            i_next = (i + 1) % segments
            
            # Primeiro triângulo da lateral
            points.extend(bottom_vertices[i])
            points.extend(top_vertices[i])
            points.extend(bottom_vertices[i_next])
            
            # Segundo triângulo da lateral
            points.extend(bottom_vertices[i_next])
            points.extend(top_vertices[i])
            points.extend(top_vertices[i_next])
        
        # Triângulos do topo (fan triangulation)
        for i in range(segments):
            i_next = (i + 1) % segments
            
            # Triângulo do topo
            points.extend(top_center)
            points.extend(top_vertices[i_next])
            points.extend(top_vertices[i])
        
        # Triângulos da base (fan triangulation)
        for i in range(segments):
            i_next = (i + 1) % segments
            
            # Triângulo da base
            points.extend(bottom_center)
            points.extend(bottom_vertices[i])
            points.extend(bottom_vertices[i_next])
        
        # Chama triangleSet para renderizar
        GL.triangleSet(points, colors)

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # Remove possíveis headlights antigos
        GL.lights = [l for l in GL.lights if l.get('type') != 'headlight']
        GL.headlight_enabled = bool(headlight)
        
        if GL.headlight_enabled:
            headlight_info = {
                'type': 'headlight',
                'ambientIntensity': 0.2,
                'color': [1.0, 1.0, 1.0],
                'intensity': 1.0,
                'direction': [0.0, 0.0, -1.0]
            }
            GL.lights.append(headlight_info)
            print(f"NavigationInfo: headlight ATIVADO")
        else:
            print(f"NavigationInfo: headlight DESATIVADO")
            
        # Debug: mostra estado das luzes
        print(f"Total de luzes ativas: {len(GL.lights)}")

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # Normaliza a direção da luz
        dir_norm = math.sqrt(sum(d*d for d in direction)) or 1.0
        normalized_direction = [d / dir_norm for d in direction]
        
        # Armazena informações da luz direcional
        light_info = {
            'type': 'directional',
            'ambientIntensity': float(ambientIntensity),
            'color': [float(c) for c in color],
            'intensity': float(intensity),
            'direction': normalized_direction
        }
        
        # Adiciona à lista de luzes (remove luzes direcionais anteriores para evitar duplicação)
        GL.lights = [l for l in GL.lights if l.get('type') != 'directional']
        GL.lights.append(light_info)
        
        # Debug: mostra configuração da luz
        print(f"DirectionalLight configurada: cor={color}, intensidade={intensity}, direção={direction}")

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("PointLight : color = {0}".format(color)) # imprime no terminal
        # print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        # print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Fog : color = {0}".format(color)) # imprime no terminal
        # print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Marca que está rodando animação para otimizações
        GL.is_animation = True
        GL.ssaa_factor = 4  # Supersampling 4x4 para qualidade balanceada

        if cycleInterval <= 0:
            return 0.0
            
        # Cria chave única para este sensor
        key = (cycleInterval, loop)
        now = time.time()
        state = GL._time_sensor_states.get(key)
        
        if state is None:
            state = {'start': now, 'cycles': 0}
            GL._time_sensor_states[key] = state
            
        elapsed = now - state['start']
        
        if loop:
            # Animação em loop
            cycle_position = elapsed % cycleInterval
            fraction = cycle_position / cycleInterval
            current_cycle = int(elapsed // cycleInterval)
            if current_cycle > state['cycles']:
                state['cycles'] = current_cycle
                print(f"TimeSensor: Ciclo {current_cycle}, fração={fraction:.3f}")
        else:
            # Animação única
            fraction = min(1.0, elapsed / cycleInterval)
            if fraction >= 1.0:
                print(f"TimeSensor: Animação finalizada em {elapsed:.2f}s")
                return 1.0
                
        return fraction

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D. Usando Catmull-Rom spline."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zero a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave.

        if not key or not keyValue or len(key) == 0:
            return [0.0, 0.0, 0.0]
        
        # Garante que set_fraction está no intervalo [0, 1]
        set_fraction = max(0.0, min(1.0, set_fraction))
        
        # Casos simples
        if len(key) == 1:
            return keyValue[0:3]
        
        # Extremos
        if set_fraction <= key[0]:
            return keyValue[0:3]
        if set_fraction >= key[-1]:
            start_idx = (len(key)-1) * 3
            return keyValue[start_idx:start_idx+3]
        
        # Encontra segmento de interpolação
        for seg in range(len(key)-1):
            k0 = key[seg]
            k1 = key[seg+1]
            if k0 <= set_fraction <= k1:
                # Parâmetro local t no segmento [0,1]
                t = (set_fraction - k0) / (k1 - k0) if k1 > k0 else 0.0
                
                # Função para obter ponto por índice
                def get_point(i):
                    i = max(0, min(i, len(key)-1))
                    base = i * 3
                    return [keyValue[base], keyValue[base+1], keyValue[base+2]]
                
                # Pontos para Catmull-Rom spline
                p1 = get_point(seg)     # ponto atual
                p2 = get_point(seg+1)   # próximo ponto
                
                # Ponto anterior
                if seg == 0:
                    p0 = get_point(len(key)-1 if closed else 0)  # primeiro ou último se fechado
                else:
                    p0 = get_point(seg-1)
                
                # Ponto seguinte
                if seg+2 >= len(key):
                    p3 = get_point(0 if closed else len(key)-1)  # primeiro se fechado
                else:
                    p3 = get_point(seg+2)
                
                # Catmull-Rom spline interpolation
                # Formula: 0.5 * (2*P1 + (-P0+P2)*t + (2*P0-5*P1+4*P2-P3)*t² + (-P0+3*P1-3*P2+P3)*t³)
                t2 = t * t
                t3 = t2 * t
                
                result = []
                for c in range(3):  # x, y, z components
                    val = 0.5 * (
                        2 * p1[c] + 
                        (-p0[c] + p2[c]) * t + 
                        (2*p0[c] - 5*p1[c] + 4*p2[c] - p3[c]) * t2 + 
                        (-p0[c] + 3*p1[c] - 3*p2[c] + p3[c]) * t3
                    )
                    result.append(val)
                
                return result
        
        return [0.0, 0.0, 0.0]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação específicos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zero a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        if not key or not keyValue or len(key) == 0:
            return [0, 0, 1, 0]  # rotação padrão (eixo Z, ângulo 0)
        
        # Garante que set_fraction está no intervalo [0, 1]
        set_fraction = max(0.0, min(1.0, set_fraction))
        
        # Caso simples: apenas uma chave
        if len(key) == 1:
            return keyValue[0:4]
        
        # Extremos
        if set_fraction <= key[0]:
            return keyValue[0:4]
        if set_fraction >= key[-1]:
            start_idx = (len(key)-1) * 4
            return keyValue[start_idx:start_idx+4]
        
        # Funções auxiliares para interpolação spherical linear (SLERP)
        def axis_angle_to_quat(axis_angle):
            """Converte eixo-ângulo para quaternion."""
            x, y, z, angle = axis_angle
            norm = math.sqrt(x*x + y*y + z*z) or 1.0
            x, y, z = x/norm, y/norm, z/norm
            half_angle = angle / 2.0
            sin_half = math.sin(half_angle)
            return [x*sin_half, y*sin_half, z*sin_half, math.cos(half_angle)]
        
        def quat_to_axis_angle(quat):
            """Converte quaternion para eixo-ângulo."""
            x, y, z, w = quat
            # Normaliza quaternion
            norm = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
            
            # Calcula ângulo
            angle = 2 * math.acos(min(1.0, abs(w)))
            sin_half = math.sqrt(1 - w*w)
            
            if sin_half < 1e-6:  # Quase sem rotação
                return [1.0, 0.0, 0.0, 0.0]
            
            return [x/sin_half, y/sin_half, z/sin_half, angle]
        
        def slerp(q1, q2, t):
            """Spherical Linear Interpolation entre dois quaternions."""
            dot = sum(q1[i] * q2[i] for i in range(4))
            
            # Se dot produto negativo, negate um quaternion para tomar caminho mais curto
            if dot < 0:
                q2 = [-c for c in q2]
                dot = -dot
            
            # Se quaternions muito próximos, usa interpolação linear
            if dot > 0.9995:
                result = [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
                # Normaliza resultado
                norm = math.sqrt(sum(r*r for r in result)) or 1.0
                return [r/norm for r in result]
            
            # Calcula ângulo entre quaternions
            theta_0 = math.acos(max(-1.0, min(1.0, abs(dot))))
            sin_theta_0 = math.sin(theta_0)
            theta = theta_0 * t
            sin_theta = math.sin(theta)
            
            s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            return [s0 * q1[i] + s1 * q2[i] for i in range(4)]
        
        # Encontra segmento de interpolação
        for seg in range(len(key) - 1):
            k0, k1 = key[seg], key[seg + 1]
            if k0 <= set_fraction <= k1:
                # Parâmetro local t no segmento [0,1]
                t = (set_fraction - k0) / (k1 - k0) if k1 > k0 else 0.0
                
                # Obtém orientações do segmento
                start_idx = seg * 4
                end_idx = (seg + 1) * 4
                orientation1 = keyValue[start_idx:start_idx+4]
                orientation2 = keyValue[end_idx:end_idx+4]
                
                # Converte para quaternions
                q1 = axis_angle_to_quat(orientation1)
                q2 = axis_angle_to_quat(orientation2)
                
                # Interpola usando SLERP
                interpolated_quat = slerp(q1, q2, t)
                
                # Converte de volta para eixo-ângulo
                return quat_to_axis_angle(interpolated_quat)
        
        return [0, 0, 1, 0]  # Fallback

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
